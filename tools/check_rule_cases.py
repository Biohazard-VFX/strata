#!/usr/bin/env python3
"""Prove each source check rejects its violation and accepts allowed code, in isolated repos."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def absolute(*parts):
    return '/' + '/'.join(parts)


BASE = {
    'runtime/src/core/unit.c': '#include <stdint.h>\n',
    'runtime/src/backend/unit.c': '#include "strata.h"\n',
    'runtime/src/cli/unit.c': '#include "strata.h"\n',
    'runtime/include/strata.h': '#include <stdint.h>\n',
    'training/strata_train/__init__.py': '',
    'contract/model.json': '{}\n',
}
CASES = [
    ('public hosts', {'hosts/nuke/unit.cpp':'#include "strata.h"\n#include <DDImage/DeepFilterOp.h>\n', 'hosts/comfyui/nodes.py':'import ctypes\n'}, None),
    ('core ORT', {'runtime/src/core/unit.c':'#include <onnxruntime_c_api.h>\n'}, 'core cannot include'),
    ('CLI private', {'runtime/src/cli/unit.c':'#include "../core/private.h"\n','runtime/src/core/private.h':''}, 'may only include strata.h'),
    ('bench private', {'runtime/src/bench/unit.c':'#include <onnxruntime_c_api.h>\n'}, 'may only include strata.h'),
    ('public header private', {'runtime/include/strata.h':'#include "../src/core/private.h"\n','runtime/src/core/private.h':''}, 'public header'),
    ('runtime Python', {'runtime/tests/unit.py':'import strata_train\n'}, 'runtime cannot depend'),
    ('training import', {'training/strata_train/unit.py':'from runtime import core\n'}, 'cross-side imports'),
    ('training dynamic import', {'training/strata_train/unit.py':'import importlib\nimportlib.import_module("runtime.core")\n'}, 'cross-side imports'),
    ('host private', {'hosts/nuke/unit.cpp':'#include "../../runtime/src/core/private.h"\n'}, 'hosts must use'),
    ('host alternate suffix', {'hosts/ofx/unit.cxx':'#include "private.h"\n'}, 'only Strata include'),
    ('host Python import', {'hosts/comfyui/nodes.py':'import runtime\n'}, 'hosts may not import'),
    ('host Python dynamic import', {'hosts/comfyui/nodes.py':'__import__("strata_train")\n'}, 'hosts may not import'),
    ('runtime C++', {'runtime/src/core/unit.cpp':'int value;\n'}, 'must be C11'),
    ('root C build', {'CMakeLists.txt':''}, 'build files belong'),
    ('datagen nonempty', {'datagen/unit.py':''}, 'datagen must remain empty'),
    ('tracked EXR', {'models/forbidden.exr':''}, 'generated artifact is tracked'),
    ('long K&R function', {'runtime/src/core/unit.c':'int f(void) {\n' + 'f();\n' * 100 + '}\n'}, 'function f exceeds 100 lines'),
    ('long function, braces in literals', {'runtime/tests/unit.c':'int g(void)\n{\n\tconst char *s = "}";\n' + '\tg(); /* } */\n' * 99 + '}\n'}, 'function g exceeds 100 lines'),
    ('short functions', {'runtime/src/core/unit.c':'static const int x[] = {\n' + '1,\n' * 120 + '};\nint f(void) {\n' + 'f();\n' * 97 + '}\n'}, None),
]
COMMENT_CASES = [
    ('C line comments', {'runtime/src/core/unit.c':'// one\n// two\nint x;\n'}, 'consecutive comment lines'),
    ('C block comment', {'runtime/tests/unit.c':'/* one\n * two */\nint x;\n'}, 'comment exceeds one line'),
    ('Python comments', {'tools/unit.py':'# one\n# two\nx = 1\n'}, 'consecutive comment lines'),
    ('training marker', {'training/strata_train/unit.py':'x = 1  # ' + 'TO' + 'DO later\n'}, 'forbidden marker'),
    ('C literal not comment', {'runtime/src/core/unit.c':'const char *s = "// a";\nconst char *t = "// b";\n/* one */\n'}, None),
]
DECISION_CASES = [
    ('wrapped entry', {'docs/DECISIONS.md':'# Decisions\n\n- one\n  continued\n'}, 'single "- " lines'),
    ('long entry', {'docs/DECISIONS.md':'# Decisions\n\n- ' + 'x' * 201 + '\n'}, 'exceeds 200 characters'),
    ('valid entries', {'docs/DECISIONS.md':'# Decisions\n\n- one\n- two\n'}, None),
]
PATH_CASES = [
    ('absolute Python', {'runtime/tests/unit.py':f'run(["{absolute("usr","bin","python3")}", "x"])\n'}, 'absolute interpreter path'),
    ('absolute shebang', {'tools/unit.sh':f'#!{absolute("bin","bash")}\n'}, 'absolute interpreter path'),
    ('home path', {'training/strata_train/unit.py':f'ROOT = "{absolute("home","user","strata")}"\n'}, 'machine-specific absolute path'),
    ('mnt path', {'runtime/README.md':f'cmake -S {absolute("mnt","work","strata")}\n'}, 'machine-specific absolute path'),
    ('portable paths', {'tools/unit.sh':'#!/usr/bin/env bash\n"$PWD/training/.venv/bin/python" -V\n',
                        'runtime/tests/unit.py':'import sys\nURL = "https://example.com/home/page"\nrun([sys.executable])\n'}, None),
]
(ROOT/'build').mkdir(exist_ok=True)


def run(name, files, error, script):
    with tempfile.TemporaryDirectory(prefix='rule-cases-',dir=ROOT/'build') as temporary:
        directory = Path(temporary)
        subprocess.run(['git','init','-qb','main',str(directory)],check=True)
        for relative,text in {**BASE,**files}.items():
            path = directory/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text)
        (directory/'tools').mkdir(exist_ok=True)
        for tool in (script,'source_scan.py'):
            shutil.copy2(ROOT/'tools'/tool,directory/'tools'/tool)
        if name == 'tracked EXR':
            subprocess.run(['git','add','--force','models/forbidden.exr'],cwd=directory,check=True)
        result = subprocess.run(['bash','tools/'+script],cwd=directory,capture_output=True,text=True)
        if error is None:
            assert result.returncode == 0, (name,result.stderr)
        else:
            assert result.returncode != 0 and error in result.stderr, (name,result.stdout,result.stderr)


for cases,script in ((CASES,'check_layers.sh'),(COMMENT_CASES,'check_comments.sh'),(DECISION_CASES,'check_decisions.sh'),(PATH_CASES,'check_paths.sh')):
    for name,files,error in cases:
        run(name,files,error,script)
print(f'rule cases: {len(CASES)} layer, {len(COMMENT_CASES)} comment, {len(DECISION_CASES)} decision and {len(PATH_CASES)} path scenarios passed')
