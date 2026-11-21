#!/usr/bin/env python3
"""Exercise layer violations and allowed public host dependencies in isolated repos."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
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
    ('CLI private', {'runtime/src/cli/unit.c':'#include "../core/private.h"\n','runtime/src/core/private.h':''}, 'CLI may only include'),
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
]
(ROOT/'build').mkdir(exist_ok=True)
for name,files,error in CASES:
    with tempfile.TemporaryDirectory(prefix='layer-rules-',dir=ROOT/'build') as temporary:
        directory = Path(temporary)
        subprocess.run(['git','init','-qb','main',str(directory)],check=True)
        for relative,text in {**BASE,**files}.items():
            path = directory/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text)
        (directory/'tools').mkdir()
        shutil.copy2(ROOT/'tools/check_layers.sh',directory/'tools/check_layers.sh')
        if name == 'tracked EXR':
            subprocess.run(['git','add','--force','models/forbidden.exr'],cwd=directory,check=True)
        result = subprocess.run(['bash','tools/check_layers.sh'],cwd=directory,capture_output=True,text=True)
        if error is None:
            assert result.returncode == 0, (name,result.stderr)
        else:
            assert result.returncode != 0 and error in result.stderr, (name,result.stdout,result.stderr)
print(f'layer rules: {len(CASES)} allowed/forbidden dependency scenarios passed')
