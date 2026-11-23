#!/usr/bin/env python3
"""Prove each source check rejects its violation and accepts allowed code, in isolated repos."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def absolute(*parts):
    return '/' + '/'.join(parts)


class Link(str):
    """A symlink to create in a rule case, pointing at this target."""


BASE = {
    'runtime/src/core/unit.c': '#include <stdint.h>\n',
    'runtime/src/backend/unit.c': '#include "strata.h"\n',
    'runtime/src/cli/unit.c': '#include "strata.h"\n',
    'runtime/include/strata.h': '#include <stdint.h>\n',
    'training/strata_train/__init__.py': '',
    'contract/strata_contract/__init__.py': '',
    'contract/dataset/strata_dataset/__init__.py': '',
    'datagen/strata_datagen/__init__.py': '',
}


def project(name, dependencies=(), sources=None):
    text = f'[project]\nname = "{name}"\ndependencies = {list(dependencies)!r}\n'.replace("'", '"')
    if sources:
        text += '[tool.uv.sources]\n' + ''.join(f'{key} = {{ path = "{value}", editable = true }}\n'
                                                for key,value in sources.items())
    return text


def lock(packages):
    text = 'version = 1\nrequires-python = ">=3.12"\n'
    for name,tables in packages.items():
        text += f'\n[[package]]\nname = "{name}"\nversion = "0"\n'
        text += 'dependencies = [' + ', '.join(f'{{ name = "{d}" }}' for d in tables.get('dependencies',())) + ']\n'
        for table in ('optional-dependencies','dev-dependencies'):
            if table in tables:
                text += f'[package.{table}]\n' + ''.join(f'{group} = [' + ', '.join(f'{{ name = "{d}" }}' for d in deps) + ']\n'
                                                       for group,deps in tables[table].items())
    return text


TRAINING_LOCK = {'strata-train':{'dependencies':['numpy','strata-contract']},'strata-contract':{},'numpy':{}}
DATAGEN_LOCK = {'strata-datagen':{'dependencies':['numpy','strata-dataset']},'strata-dataset':{'dependencies':['numpy']},'numpy':{}}
REVIEW_DATAGEN = '''[project]
name = "strata-datagen"
dependencies = ["numpy", "strata-dataset"]

[tool.uv.sources]
strata-dataset = { path = "../contract/dataset", editable = true }
strata-contract = [{ path = "../contract", editable = true }]

[dependency-groups]
dev = ["ruff", "strata-contract"]
'''
OWN_DIRECTORY_SCRIPT = '''import importlib
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
creature = importlib.import_module(f"infinigen.assets.creatures.{sys.argv[1]}")
jit = import_module("numba").njit
spec = importlib.util.find_spec(name="strata_dataset.layers")
files = importlib.resources.files("strata_dataset")
clean = {k: v for k, v in {}.items() if k != "PYTHONPATH"}
'''


BASE.update({
    '.gitignore': (ROOT/'.gitignore').read_text(),
    'training/uv.lock': lock(TRAINING_LOCK),
    'datagen/uv.lock': lock(DATAGEN_LOCK),
    'training/posttrain/uv.lock': lock({'strata-training':{'dependencies':['strata-dataset']},'strata-dataset':{}}),
    'contract/pyproject.toml': project('strata-contract'),
    'contract/dataset/pyproject.toml': project('strata-dataset',['numpy>=2.0']),
    'training/pyproject.toml': project('strata-train',['numpy','strata-contract'],{'strata-contract':'../contract'}),
    'training/posttrain/pyproject.toml': project('strata-training',['strata-dataset'],{'strata-dataset':'../../contract/dataset'}),
    'datagen/pyproject.toml': project('strata-datagen',['numpy','strata-dataset'],{'strata-dataset':'../contract/dataset'}),
})
CASES = [
    ('GT scale flag', {'training/posttrain/src/unit.py': 'flag = "--scale-mode"\n'}, 'obsolete training reference'),
    ('GT aligned variant', {'training/posttrain/src/unit.py': 'name = "oracle-aligned"\n'}, 'obsolete training reference'),
    ('GT decoder pass', {'training/posttrain/src/unit.py': 'def _decoder_oracle_scale(model): return 1\n'}, 'obsolete training reference'),
    ('duplicate metric factor', {'training/posttrain/src/unit.py': 'key = "raw_metric_scale"\n'}, 'obsolete training reference'),
    ('GT scale function', {'training/posttrain/src/unit.py': 'def oracle_metric_scale(data, inverse): return 1\n'}, 'obsolete training reference'),
    ('GT scale option', {'training/posttrain/configs/neutral.json': '{"scale_mode": "raw"}'}, 'obsolete training reference'),
    ('GT scale metric', {'training/posttrain/src/unit.py': 'def scale_align(inverse, truth): return inverse\n'}, 'obsolete training reference'),
    ('GT affine metric', {'training/posttrain/src/unit.py': 'def affine_align(inverse, truth): return inverse\n'}, 'obsolete training reference'),
    ('duplicate depth cache', {'training/posttrain/src/unit.py': 'key = "raw_inverse_depth"\n'}, 'obsolete training reference'),
    ('model metric depth', {'training/posttrain/src/unit.py': 'key = "inverse_depth"\n'}, None),
    ('versioned training prose', {'training/posttrain/src/unit.py': 'label = "v004 frozen probe"\n'}, 'obsolete training reference'),
    ('obsolete training config content', {'training/posttrain/configs/neutral.json': '{"masks": "legacy masks"}'}, 'obsolete training reference'),
    ('nested versioned config content', {'training/posttrain/configs/nested/neutral.json': '{"baseline": "V004"}'}, 'obsolete training reference'),
    ('legacy training input', {'training/posttrain/src/unit.py': 'value = entry["legacy_data"]\n'}, 'legacy training inputs'),
    ('legacy training reader', {'training/posttrain/src/unit.py': 'def read_deep(path): pass\n'}, 'legacy training readers'),
    ('legacy RGB option', {'training/posttrain/src/unit.py': 'def read_rgb(path, *, legacy=False): pass\n'}, 'legacy training readers'),
    ('legacy training config', {'training/posttrain/configs/h_head.json': '{}'}, 'legacy training configs'),
    ('legacy evaluation split', {'training/eval/v004_split.json': '{}'}, 'legacy evaluation split'),
    ('README image', {'README.md':'![x](docs/images/a.jpg)\n','docs/images/a.jpg':'x'}, None),
    ('unreferenced docs image', {'README.md':'# x\n','docs/images/a.jpg':'x'}, 'README images'),
    ('docs image wrong type', {'README.md':'![x](docs/images/a.gif)\n','docs/images/a.gif':'x'}, 'README images'),
    ('docs notes', {'README.md':'# x\n','docs/notes.md':'x'}, 'README images'),
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
    ('datagen uses dataset contract', {'datagen/strata_datagen/unit.py':'import numpy\nfrom strata_dataset import samples\n',
                                       'training/strata_train/unit.py':'from strata_dataset.layers import Layers\n'}, None),
    ('datagen imports post-training', {'datagen/strata_datagen/unit.py':'from strata_training import config\n'}, 'datagen may depend only'),
    ('dataset imports post-training', {'contract/dataset/strata_dataset/unit.py':'import strata_training\n'}, 'dataset contract imports no'),
    ('datagen imports training', {'datagen/strata_datagen/unit.py':'from strata_train import export\n'}, 'datagen may depend only'),
    ('datagen imports model contract', {'datagen/tests/unit.py':'import strata_contract\n'}, 'datagen may depend only'),
    ('datagen dynamic runtime import', {'datagen/tools/unit.py':'__import__("runtime")\n'}, 'datagen may depend only'),
    ('datagen training path', {'datagen/strata_datagen/unit.py':'ROOT = "../training/strata_train"\n'}, 'forbidden cross-side path'),
    ('training imports datagen', {'training/strata_train/unit.py':'import strata_datagen.cli\n'}, 'cross-side imports'),
    ('training tests import datagen', {'training/tests/unit.py':'from datagen import cli\n'}, 'cross-side imports'),
    ('runtime names datagen', {'runtime/tests/unit.py':'import strata_datagen\n'}, 'runtime cannot depend'),
    ('dataset contract imports datagen', {'contract/dataset/strata_dataset/unit.py':'from strata_datagen import rng\n'}, 'dataset contract imports no'),
    ('model contract imports dataset', {'contract/strata_contract/unit.py':'import strata_dataset\n'}, 'cross-side imports'),
    ('datagen requires training', {'datagen/pyproject.toml':project('strata-datagen',['strata-dataset','strata-train'],{'strata-dataset':'../contract/dataset'})}, 'may not depend on strata-train'),
    ('datagen sources model contract', {'datagen/pyproject.toml':project('strata-datagen',['strata-dataset'],{'strata-dataset':'../contract'})}, 'may not source strata-dataset'),
    ('training requires datagen', {'training/pyproject.toml':project('strata-train',['strata-contract','strata-datagen'],{'strata-contract':'../contract'})}, 'may not depend on strata-datagen'),
    ('dataset distribution collides', {'contract/dataset/pyproject.toml':project('strata-contract')}, 'distribution must be strata-dataset'),
    ('datagen conftest keyword import_module', {'datagen/conftest.py':'"""Fixtures."""\nimport importlib\nimportlib.import_module(name="strata_contract")\n'}, 'datagen may depend only'),
    ('datagen aliased import_module', {'datagen/tools/unit.py':'from importlib import import_module as load\nload("strata_contract")\n'}, 'datagen may depend only'),
    ('datagen relative dynamic name', {'datagen/strata_datagen/unit.py':'import importlib\nimportlib.import_module(".strata_contract", "contract")\n'}, 'datagen may depend only'),
    ('datagen dotted dynamic prefix', {'datagen/tests/unit.py':'import importlib as il\nname = "x"\nil.import_module(f"strata_train.{name}")\n'}, 'datagen may depend only'),
    ('datagen keyword __import__', {'datagen/tests/unit.py':'__import__(name="strata_train")\n'}, 'datagen may depend only'),
    ('training keyword find_spec', {'training/tests/unit.py':'import importlib.util\nimportlib.util.find_spec(name="strata_datagen")\n'}, 'cross-side imports'),
    ('datagen non-literal dynamic import', {'datagen/strata_datagen/unit.py':'import importlib\nimportlib.import_module("strata_" + "x".upper())\n'}, 'literal top-level package'),
    ('datagen undotted dynamic prefix', {'datagen/strata_datagen/unit.py':'from importlib import import_module\nname = "x"\nimport_module(f"strata_contract{name}")\n'}, 'literal top-level package'),
    ('datagen relative dynamic without package', {'datagen/strata_datagen/unit.py':'import importlib\nimportlib.import_module(".x")\n'}, 'literal name and package'),
    ('datagen importer by reference', {'datagen/strata_datagen/unit.py':'import importlib\nload = importlib.import_module\nload("strata_contract")\n'}, 'without a direct call'),
    ('datagen importlib by getattr', {'datagen/strata_datagen/unit.py':'import importlib\ngetattr(importlib, "import_module")("strata_contract")\n'}, 'importlib used indirectly'),
    ('datagen path loader', {'datagen/tests/unit.py':'from importlib.util import spec_from_file_location\n'}, 'unsupported import machinery importlib.util.spec_from_file_location'),
    ('datagen runpy', {'datagen/tests/unit.py':'import runpy\nrunpy.run_path("x.py")\n'}, 'unsupported import machinery runpy'),
    ('datagen site', {'datagen/conftest.py':'import site\nsite.addsitedir("x")\n'}, 'unsupported import machinery site'),
    ('datagen sys.path append', {'datagen/tests/unit.py':'import sys\nsys.path.append("x")\n'}, 'sys.path changes where imports resolve'),
    ('datagen imported sys.path', {'datagen/tests/unit.py':'from sys import path\npath.insert(0, "x")\n'}, 'sys.path changes where imports resolve'),
    ('training meta_path', {'training/strata_train/unit.py':'import sys\nsys.meta_path.clear()\n'}, 'sys.meta_path changes'),
    ('datagen PYTHONPATH environment', {'datagen/tests/unit.py':'import os\nenv = {**os.environ, "PYTHONPATH": "x"}\n'}, 'PYTHONPATH changes'),
    ('datagen PYTHONPATH keyword', {'datagen/tests/unit.py':'env = dict(PYTHONPATH="x")\n'}, 'PYTHONPATH changes'),
    ('relative import outside a package', {'datagen/conftest.py':'from . import strata_datagen\n'}, 'relative import leaves its package'),
    ('relative import above its package', {'datagen/strata_datagen/unit.py':'from .. import training\n'}, 'relative import leaves its package'),
    ('own directory and rooted dynamic imports', {'datagen/blender/unit.py':OWN_DIRECTORY_SCRIPT,
                                                  'datagen/strata_datagen/unit.py':'from . import other\nfrom .other import x\n'}, None),
    ('other directory on sys.path', {'datagen/blender/unit.py':OWN_DIRECTORY_SCRIPT.replace('.parent))','.parents[2]))')}, 'sys.path changes'),
    ('own directory outside Blender scripts', {'datagen/strata_datagen/unit.py':OWN_DIRECTORY_SCRIPT}, 'sys.path changes'),
    ('own directory in a Blender subdirectory', {'datagen/blender/sub/unit.py':OWN_DIRECTORY_SCRIPT}, 'sys.path changes'),
    ('exec imports model contract', {'datagen/strata_datagen/unit.py':'exec("import strata_contract")\n'}, 'code string: datagen may depend only'),
    ('eval dynamic import', {'datagen/tests/unit.py':'eval(\'__import__("strata_contract")\')\n'}, 'code string: datagen may depend only'),
    ('compile from import', {'datagen/tools/unit.py':'code = compile(source="from strata_train import x", filename="f", mode="exec")\n'}, 'code string: datagen may depend only'),
    ('builtins exec bytes', {'training/strata_train/unit.py':'import builtins\nbuiltins.exec(b"import strata_datagen")\n'}, 'code string: cross-side imports'),
    ('code strings without forbidden imports', {'datagen/strata_datagen/unit.py':'import re\nexec("import numpy")\nre.compile("import strata_contract")\n'}, None),
    ('review: dev group and list source', {'datagen/pyproject.toml':REVIEW_DATAGEN}, 'may not depend on strata-contract (dependency-groups.dev)'),
    ('list-valued forbidden source', {'datagen/pyproject.toml':REVIEW_DATAGEN.replace(', "strata-contract"]',']')}, 'may not source strata-contract'),
    ('optional extra requires training', {'datagen/pyproject.toml':project('strata-datagen',['strata-dataset'],{'strata-dataset':'../contract/dataset'}) + '[project.optional-dependencies]\nfit = ["strata-train>=0"]\n'}, 'may not depend on strata-train (optional-dependencies.fit)'),
    ('uv dev-dependencies require contract', {'datagen/pyproject.toml':project('strata-datagen',['strata-dataset'],{'strata-dataset':'../contract/dataset'}) + '[tool.uv]\ndev-dependencies = ["Strata_Contract"]\n'}, 'may not depend on strata-contract (tool.uv.dev-dependencies)'),
    ('local path under another name', {'datagen/pyproject.toml':project('strata-datagen',['numpy','strata-dataset'],{'strata-dataset':'../contract/dataset','numpy':'../contract'})}, 'may not source numpy'),
    ('list-valued local path under another name', {'datagen/pyproject.toml':project('strata-datagen',['numpy','strata-dataset'],{'strata-dataset':'../contract/dataset'}).replace('[tool.uv.sources]\n','[tool.uv.sources]\nnumpy = [{ path = "../contract", marker = "sys_platform == \'linux\'" }]\n')}, 'may not source numpy'),
    ('list-valued allowed source', {'datagen/pyproject.toml':project('strata-datagen',['strata-dataset']) + '[tool.uv.sources]\nstrata-dataset = [{ path = "../contract/dataset", editable = true }]\n'}, None),
    ('Strata package from a registry', {'datagen/pyproject.toml':project('strata-datagen',['strata-dataset'])}, 'strata-dataset must come from the local path'),
    ('Strata package from git', {'datagen/pyproject.toml':project('strata-datagen',['strata-dataset']) + '[tool.uv.sources]\nstrata-dataset = { git = "https://example.com/strata" }\n'}, 'may not source strata-dataset'),
    ('training may use both contracts', {'training/pyproject.toml':project('strata-train',['strata-contract'],{'strata-contract':'../contract','strata-dataset':'../contract/dataset'}) + '[dependency-groups]\ndev = ["strata-dataset"]\n',
                                         'training/uv.lock':lock({**TRAINING_LOCK,'strata-dataset':{}})}, None),
    ('datagen lock dev group holds model contract', {'datagen/uv.lock':lock({**DATAGEN_LOCK,'strata-datagen':{'dependencies':['strata-dataset'],'dev-dependencies':{'dev':['strata-contract']}},'strata-contract':{}})}, 'strata-datagen environment may not contain strata-contract'),
    ('dataset contract extra reaches training', {'training/uv.lock':lock({**TRAINING_LOCK,'strata-train':{'dependencies':['strata-contract','strata-dataset']},'strata-dataset':{'optional-dependencies':{'fit':['strata-train']}}})}, 'strata-dataset environment may not contain strata-train'),
    ('lock without its project', {'datagen/uv.lock':lock({'numpy':{}})}, 'lock does not contain its project strata-datagen'),
    ('gitignore misses symlinks', {'.gitignore':'build/\n.venv/\n'}, '.gitignore must ignore build as a directory or a symlink'),
    ('tracked EXR', {'models/forbidden.exr':''}, 'generated artifact is tracked'),
    ('long K&R function', {'runtime/src/core/unit.c':'int f(void) {\n' + 'f();\n' * 100 + '}\n'}, 'function f exceeds 100 lines'),
    ('long function, braces in literals', {'runtime/tests/unit.c':'int g(void)\n{\n\tconst char *s = "}";\n' + '\tg(); /* } */\n' * 99 + '}\n'}, 'function g exceeds 100 lines'),
    ('short functions', {'runtime/src/core/unit.c':'static const int x[] = {\n' + '1,\n' * 120 + '};\nint f(void) {\n' + 'f();\n' * 97 + '}\n'}, None),
]
COMMENT_CASES = [
    ('C line comments', {'runtime/src/core/unit.c':'// one\n// two\nint x;\n'}, 'consecutive comment lines'),
    ('C block comment', {'runtime/tests/unit.c':'/* one\n * two */\nint x;\n'}, 'comment exceeds one line'),
    ('Python comments', {'tools/unit.py':'# one\n# two\nx = 1\n'}, 'consecutive comment lines'),
    ('datagen comments', {'datagen/strata_datagen/unit.py':'# one\n# two\nx = 1\n'}, 'consecutive comment lines'),
    ('training marker', {'training/strata_train/unit.py':'x = 1  # ' + 'TO' + 'DO later\n'}, 'forbidden marker'),
    ('C literal not comment', {'runtime/src/core/unit.c':'const char *s = "// a";\nconst char *t = "// b";\n/* one */\n'}, None),
]
DECISION_CASES = [
    ('wrapped entry', {'docs/DECISIONS.md':'# Decisions\n\n- one\n  continued\n'}, 'single "- " lines'),
    ('long entry', {'docs/DECISIONS.md':'# Decisions\n\n- ' + 'x' * 201 + '\n'}, 'exceeds 200 characters'),
    ('datagen wrapped entry', {'datagen/DECISIONS.md':'# Decisions\n\n- one\n  continued\n'}, 'datagen/DECISIONS.md:4'),
    ('datagen long entry', {'datagen/DECISIONS.md':'# Decisions\n\n- ' + 'x' * 201 + '\n'}, 'datagen/DECISIONS.md:3: entry exceeds'),
    ('datagen wrong title', {'datagen/DECISIONS.md':'# Datagen decisions\n\n- one\n'}, 'datagen/DECISIONS.md:1: must start'),
    ('datagen valid entries', {'datagen/DECISIONS.md':'# Decisions\n\n- one\n'}, None),
    ('valid entries', {'docs/DECISIONS.md':'# Decisions\n\n- one\n- two\n'}, None),
]
PATH_CASES = [
    ('absolute Python', {'runtime/tests/unit.py':f'run(["{absolute("usr","bin","python3")}", "x"])\n'}, 'absolute interpreter path'),
    ('absolute shebang', {'tools/unit.sh':f'#!{absolute("bin","bash")}\n'}, 'absolute interpreter path'),
    ('home path', {'training/strata_train/unit.py':f'ROOT = "{absolute("home","user","strata")}"\n'}, 'machine-specific absolute path'),
    ('mnt path', {'runtime/README.md':f'cmake -S {absolute("mnt","work","strata")}\n'}, 'machine-specific absolute path'),
    ('portable paths', {'tools/unit.sh':'#!/usr/bin/env bash\n"$PWD/training/.venv/bin/python" -V\n',
                        'runtime/tests/unit.py':'import sys\nURL = "https://example.com/home/page"\nrun([sys.executable])\n'}, None),
    ('absolute symlink', {'docs/notes.md':Link(absolute('etc','hostname'))}, 'docs/notes.md: symlink target is absolute'),
    ('escaping symlink', {'training/data':Link('../../outside/data')}, 'training/data: symlink target leaves the repository'),
    ('symlink escaping through another symlink', {'docs/root':Link('..'),'indirect':Link('docs/root/../outside')},
     'indirect: symlink target leaves the repository'),
    ('dangling symlink', {'docs/notes.md':Link('missing.md')}, 'docs/notes.md: symlink target is missing or loops'),
    ('symlink loop', {'docs/a':Link('b'),'docs/b':Link('a')}, 'symlink target is missing or loops'),
    ('symlinks inside the repository', {'AGENTS.md':'rules\n','CLAUDE.md':Link('AGENTS.md'),
                                        'datagen/tests/data/rules.md':Link('../../../AGENTS.md')}, None),
]
(ROOT/'build').mkdir(exist_ok=True)


def run(name, files, error, script):
    with tempfile.TemporaryDirectory(prefix='rule-cases-',dir=ROOT/'build') as temporary:
        directory = Path(temporary)
        subprocess.run(['git','init','-qb','main',str(directory)],check=True)
        for relative,text in {**BASE,**files}.items():
            path = directory/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            if isinstance(text,Link):
                path.symlink_to(text)
            else:
                path.write_text(text)
        (directory/'tools').mkdir(exist_ok=True)
        for tool in (script,'source_scan.py','boundaries.py'):
            shutil.copy2(ROOT/'tools'/tool,directory/'tools'/tool)
        if name == 'tracked EXR':
            subprocess.run(['git','add','--force','models/forbidden.exr'],cwd=directory,check=True)
        if any(relative.startswith('docs/') for relative in files):
            subprocess.run(['git','add','--force','--','docs'],cwd=directory,check=True)
        result = subprocess.run(['bash','tools/'+script],cwd=directory,capture_output=True,text=True)
        if error is None:
            assert result.returncode == 0, (name,result.stderr)
        else:
            assert result.returncode != 0 and error in result.stderr, (name,result.stdout,result.stderr)


ENVIRONMENT_CASES = [
    ('clean environments', {}, None),
    ('training holds both contracts', {'training':{'strata_contract/__init__.py':'','strata_dataset/__init__.py':''}}, None),
    ('datagen finds model contract', {'datagen':{'strata_contract/__init__.py':''}}, 'datagen environment'),
    ('datagen path file reaches model contract', {'datagen':{'escape.pth':'{root}/contract\n'},
                                                  'repo':{'contract/strata_contract/__init__.py':''}}, 'can import strata_contract'),
    ('datagen holds training distribution', {'datagen':{'strata_train-0.1.0.dist-info/METADATA':'Metadata-Version: 2.1\nName: strata-train\nVersion: 0.1.0\n'}}, 'holds strata-train'),
    ('post-training finds datagen', {'training/posttrain':{'strata_datagen.py':''}}, 'training/posttrain environment'),
    ('datagen finds post-training', {'datagen':{'strata_training.py':''}}, 'can import strata_training'),
    ('post-training holds model contract', {'training/posttrain':{'strata_contract-1.0.dist-info/METADATA':'Metadata-Version: 2.1\nName: strata-contract\nVersion: 1.0\n'}}, 'holds strata-contract'),
    ('training finds datagen', {'training':{'strata_datagen.py':''}}, 'training environment'),
]


def run_environment(name, files, error):
    with tempfile.TemporaryDirectory(prefix='environment-cases-',dir=ROOT/'build') as temporary:
        directory = Path(temporary)
        (directory/'tools').mkdir()
        for tool in ('check_environments.py','boundaries.py'):
            shutil.copy2(ROOT/'tools'/tool,directory/'tools'/tool)
        for relative,text in files.get('repo',{}).items():
            (directory/relative).parent.mkdir(parents=True,exist_ok=True)
            (directory/relative).write_text(text)
        for project in ('training','training/posttrain','datagen'):
            venv = directory/project/'.venv'
            subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],check=True)
            site = subprocess.check_output([str(venv/'bin'/'python'),'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True)
            for relative,text in files.get(project,{}).items():
                path = Path(site.strip())/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(text.replace('{root}',str(directory)))
        env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
        result = subprocess.run([str(directory/'training'/'.venv'/'bin'/'python'),'tools/check_environments.py'],
                                cwd=directory,capture_output=True,text=True,env=env)
        if error is None:
            assert result.returncode == 0, (name,result.stderr)
        else:
            assert result.returncode != 0 and error in result.stderr, (name,result.stdout,result.stderr)


for name,files,error in ENVIRONMENT_CASES:
    run_environment(name,files,error)
for cases,script in ((CASES,'check_layers.sh'),(COMMENT_CASES,'check_comments.sh'),(DECISION_CASES,'check_decisions.sh'),(PATH_CASES,'check_paths.sh')):
    for name,files,error in cases:
        run(name,files,error,script)
print(f'rule cases: {len(ENVIRONMENT_CASES)} environment, {len(CASES)} layer, {len(COMMENT_CASES)} comment, {len(DECISION_CASES)} decision and {len(PATH_CASES)} path scenarios passed')
