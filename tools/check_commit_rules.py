#!/usr/bin/env python3
"""Prove message grammar, range traversal and the installed hook in an isolated repo."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('rules',ROOT/'tools/check_commits.py')
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)
for message in (
    'feat: add depth', 'Feat: case-insensitive type', 'FIX(API): correct case',
    'docs: grammar\n\nBREAKING CHANGE examples are ordinary prose.',
    'fix(runtime): reject bad input', 'refactor!: change API',
    'feat(core)!: change units\n\nBREAKING CHANGE: depths are metres\n',
    'docs: usage\n\nA free-form body.\n\nRefs: #1', 'perf: reduce allocations',
):
    assert rules.validate(message) is None, message
for message in (
    'plain message','feat( ): blank scope','feat(): empty scope','feat: ',
    'feat(core) missing colon','fix: message\nbody without separator',
    'feat: change\n\nBREAKING CHANGE:',
):
    assert rules.validate(message) is not None, message
(ROOT/'build').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='commit-rules-',dir=ROOT/'build') as temporary:
    directory = Path(temporary)
    subprocess.run(['git','init','-q',str(directory)],check=True)
    subprocess.run(['git','config','user.name','Commit Rule Test'],cwd=directory,check=True)
    subprocess.run(['git','config','user.email','commit-test@example.invalid'],cwd=directory,check=True)
    for name in ('check_commits.py','check_commits.sh','install_hooks.sh','hooks/commit-msg'):
        path = directory/'tools'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'tools'/name,path)
    subprocess.run(['bash','tools/install_hooks.sh'],cwd=directory,check=True)
    (directory/'file').write_text('fixture\n')
    subprocess.run(['git','add','file'],cwd=directory,check=True)
    rejected = subprocess.run(['git','commit','-qm','bad message'],cwd=directory,capture_output=True,text=True)
    assert rejected.returncode != 0 and 'commit message:' in rejected.stderr
    subprocess.run(['git','commit','-qm','chore: initialize fixture'],cwd=directory,check=True)
    (directory/'file').write_text('changed\n')
    subprocess.run(['git','add','file'],cwd=directory,check=True)
    subprocess.run(['git','commit','-qm','test(hook): accept conventional message'],cwd=directory,check=True)
    subprocess.run(['bash','tools/check_commits.sh','HEAD~1..HEAD'],cwd=directory,check=True)
    subprocess.run(['git','-c','core.hooksPath=/dev/null','commit','--allow-empty','-qm','invalid fixture commit'],cwd=directory,check=True)
    rejected = subprocess.run(['bash','tools/check_commits.sh','HEAD~1..HEAD'],cwd=directory,capture_output=True,text=True)
    assert rejected.returncode != 0 and 'expected type' in rejected.stderr
    subprocess.run(['git','commit','--allow-empty','-qm','test: accept last commit'],cwd=directory,check=True)
    rejected = subprocess.run(['bash','tools/check_commits.sh','HEAD~2..HEAD'],cwd=directory,capture_output=True,text=True)
    assert rejected.returncode != 0 and 'invalid fixture commit' in rejected.stderr
print('Conventional Commit grammar, history range, rejection and installed hook passed')
