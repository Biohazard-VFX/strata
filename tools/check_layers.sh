#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from pathlib import Path
import re
import sys
import subprocess

sys.path.insert(0, 'tools')
from boundaries import LAYERS, PROJECTS, import_violations, lock_violations, project_violations, python_files
from source_scan import c_functions

errors = []
tracked = subprocess.check_output(['git','ls-files'],text=True).splitlines()
for name in tracked:
    path = Path(name)
    if any(part in ('build','models','.venv','__pycache__') for part in path.parts) or path.suffix in ('.exr','.onnx','.npz','.weights','.pt','.log'):
        errors.append(f'generated artifact is tracked: {name}')
    if name.startswith('docs/') and path.name not in ('DECISIONS.md','BUILD.md','USAGE.md'):
        errors.append(f'docs may only contain decisions and build/usage: {name}')
for name in ('runtime/src/core','runtime/src/backend','runtime/src/cli','runtime/include','training/strata_train',
             'contract/strata_contract','contract/dataset/strata_dataset','datagen/strata_datagen'):
    if not Path(name).is_dir() or not any(Path(name).iterdir()):
        errors.append(f'missing architecture directory: {name}')
if Path('CMakeLists.txt').exists() or Path('CMakePresets.json').exists():
    errors.append('C build files belong under runtime/')
standard = {'assert.h','ctype.h','errno.h','float.h','inttypes.h','limits.h',
            'math.h','stdbool.h','stddef.h','stdint.h','stdio.h','stdlib.h',
            'string.h','time.h'}
for path in Path('runtime').rglob('*'):
    if not path.is_file() or path.is_relative_to('runtime/third_party'):
        continue
    if path.suffix == '.C' or path.suffix.lower() in ('.cpp','.cc','.cxx','.hpp','.hh','.hxx','.cu','.m','.mm'):
        errors.append(f'{path}: runtime source must be C11')
        continue
    if path.suffix not in ('.c','.h','.py','.txt','.json','.sh'):
        continue
    text = path.read_text()
    if re.search(r'(?:training/|datagen/|strata_train|strata_datagen)',text):
        errors.append(f'{path}: runtime cannot depend on the other side')
    if path.suffix not in ('.c','.h'):
        continue
    for match in re.finditer(r'^\s*#\s*include\s*[<"]([^>"\n]+)',text,re.M):
        header = match[1]
        if path.is_relative_to('runtime/include') and header not in standard:
            errors.append(f'{path}: public header may only include C standard headers: {header}')
        if path.is_relative_to('runtime/src/core') and ('onnxruntime' in header.lower() or 'backend' in header.lower()):
            errors.append(f'{path}: core cannot include {header}')
        if (path.is_relative_to('runtime/src/cli') or path.is_relative_to('runtime/src/bench')) and header != 'strata.h' and header not in standard:
            errors.append(f'{path}: CLI and bench may only include strata.h and C standard headers: {header}')
    for match in re.finditer(r'\btypedef\s+struct\b[^;]*;',text,re.S):
        if match[0] != 'typedef struct strata_context strata_context;':
            errors.append(f'{path}: only opaque strata_context may be typedefed')
    for name,first,last in c_functions(text):
        if last-first+1 > 100:
            errors.append(f'{path}:{first}: function {name} exceeds 100 lines')
for directory,paths in (('training',r'(?:\.\./runtime|\.\./datagen|runtime/|datagen/)'),
                        ('contract/strata_contract',r'(?:\.\./runtime|\.\./datagen|runtime/|datagen/)'),
                        ('contract/dataset',r'(?:\.\./runtime|\.\./training|\.\./datagen|runtime/|training/|datagen/)'),
                        ('datagen',r'(?:\.\./runtime|\.\./training|runtime/|training/|strata_train)')):
    for path in python_files(directory):
        if re.search(paths,path.read_text()):
            errors.append(f'{path}: forbidden cross-side path')
for directory,forbidden,message in LAYERS:
    for path in python_files(directory):
        errors += import_violations(path,forbidden,message)
for directory in PROJECTS:
    errors += project_violations(directory)
errors += lock_violations()
for path in Path('hosts').rglob('*'):
    if not path.is_file() or path.suffix.lower() not in ('.c','.h','.cc','.cpp','.cxx','.hpp','.hh','.hxx','.m','.mm','.py','.cmake','.txt'):
        continue
    text = path.read_text()
    if re.search(r'(?:runtime/src|training/|datagen/|strata_train)',text):
        errors.append(f'{path}: hosts must use the public C API only')
    for match in re.finditer(r'^\s*#\s*include\s*([<"])([^>"\n]+)',text,re.M):
        kind,header = match.groups()
        private = any(part in ('runtime','core','backend','cli','training','datagen','contract') for part in Path(header).parts)
        if (kind == '"' or private or 'strata' in header.lower() or 'onnxruntime' in header.lower()) and header != 'strata.h':
            errors.append(f'{path}: the only Strata include permitted in hosts is strata.h: {header}')
if errors:
    sys.exit('\n'.join(errors))
print('architecture directories, dependency layers, opaque handles and function sizes passed')
PY
