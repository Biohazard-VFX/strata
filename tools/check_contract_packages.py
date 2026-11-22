#!/usr/bin/env python3
"""Install both contract distributions into one environment and prove each import resolves to its own tree."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PROBE = '''
import importlib.metadata, json, strata_contract, strata_dataset
print(json.dumps({
    "strata_contract": list(strata_contract.__path__),
    "strata_dataset": list(strata_dataset.__path__),
    "distributions": sorted(d.metadata["Name"] for d in importlib.metadata.distributions()
                            if d.metadata["Name"].startswith("strata")),
}))
'''


def main():
    (ROOT/'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='contract-packages-',dir=ROOT/'build') as temporary:
        environment = Path(temporary)/'venv'
        env = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH','VIRTUAL_ENV')}
        subprocess.run(['uv','venv','--quiet','--python',sys.executable,str(environment)],check=True,env=env)
        python = environment/'bin'/'python'
        subprocess.run(['uv','pip','install','--quiet','--python',str(python),'--no-deps',
                        '-e',str(ROOT/'contract'),'-e',str(ROOT/'contract'/'dataset')],check=True,env=env)
        result = subprocess.run([str(python),'-I','-c',PROBE],capture_output=True,text=True,check=True,
                                cwd=temporary,env=env)
    found = json.loads(result.stdout)
    assert found['distributions'] == ['strata-contract','strata-dataset'], found
    assert [Path(p).resolve() for p in found['strata_contract']] == [ROOT/'contract'/'strata_contract'], found
    assert [Path(p).resolve() for p in found['strata_dataset']] == [ROOT/'contract'/'dataset'/'strata_dataset'], found
    print('strata-contract and strata-dataset install together and resolve to their own directories')


if __name__ == '__main__':
    main()
