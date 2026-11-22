#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import ast
from pathlib import Path
import re
import sys
import subprocess
import tomllib

sys.path.insert(0, 'tools')
from source_scan import c_functions

errors = []
def imported_modules(node):
    if isinstance(node,ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node,ast.ImportFrom):
        return [node.module] if node.module else [alias.name for alias in node.names]
    if isinstance(node,ast.Call) and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
        function = node.func
        if (isinstance(function,ast.Name) and function.id == '__import__') or (isinstance(function,ast.Attribute) and function.attr == 'import_module'):
            return [node.args[0].value]
    return []

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
python_layers = (
    ('training', ('runtime','datagen','strata_datagen'),
     r'(?:\.\./runtime|\.\./datagen|runtime/|datagen/)', 'cross-side imports must use contract'),
    ('contract/strata_contract', ('runtime','training','datagen','strata_train','strata_datagen','strata_dataset'),
     r'(?:\.\./runtime|\.\./datagen|runtime/|datagen/)', 'cross-side imports must use contract'),
    ('contract/dataset', ('runtime','training','datagen','contract','strata_train','strata_datagen','strata_contract'),
     r'(?:\.\./runtime|\.\./training|\.\./datagen|runtime/|training/|datagen/)',
     'the dataset contract imports no Strata layer'),
    ('datagen', ('runtime','training','contract','strata_train','strata_contract'),
     r'(?:\.\./runtime|\.\./training|runtime/|training/|strata_train)',
     'datagen may depend only on the dataset contract'),
)
for directory,forbidden,paths,message in python_layers:
    for path in Path(directory).rglob('*.py'):
        if 'typings' in path.parts or '.venv' in path.parts:
            continue
        text = path.read_text()
        if re.search(paths,text):
            errors.append(f'{path}: forbidden cross-side path')
        for node in ast.walk(ast.parse(text)):
            modules = imported_modules(node)
            if any(m.split('.')[0] in forbidden for m in modules):
                errors.append(f'{path}:{node.lineno}: {message}')
def project(path):
    data = tomllib.loads(Path(path).read_text())
    requirements = [re.split(r'[\s<>=!~;\[]',r,maxsplit=1)[0].lower() for r in data['project'].get('dependencies',[])]
    sources = data.get('tool',{}).get('uv',{}).get('sources',{})
    local = [s['path'] for s in sources.values() if isinstance(s,dict) and 'path' in s]
    return data['project']['name'],requirements,local
for path,name,allowed in (('contract/pyproject.toml','strata-contract',{}),
                          ('contract/dataset/pyproject.toml','strata-dataset',{}),
                          ('training/pyproject.toml','strata-train',{'strata-contract':'../contract'}),
                          ('datagen/pyproject.toml','strata-datagen',{'strata-dataset':'../contract/dataset'})):
    if not Path(path).is_file():
        errors.append(f'missing project: {path}')
        continue
    actual,requirements,local = project(path)
    if actual != name:
        errors.append(f'{path}: distribution must be {name}, not {actual}')
    for requirement in requirements:
        if requirement.startswith('strata') and requirement not in allowed:
            errors.append(f'{path}: {name} may not depend on {requirement}')
    if sorted(local) != sorted(allowed.values()):
        errors.append(f'{path}: local sources must be exactly {sorted(allowed.values())}, not {sorted(local)}')
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
    if path.suffix == '.py':
        for node in ast.walk(ast.parse(text)):
            modules = imported_modules(node)
            if any(m.split('.')[0] in ('runtime','training','datagen','strata_train','strata_datagen','contract','strata_contract','strata_dataset') for m in modules):
                errors.append(f'{path}:{node.lineno}: hosts may not import implementation layers')
if errors:
    sys.exit('\n'.join(errors))
print('architecture directories, dependency layers, opaque handles and function sizes passed')
PY
