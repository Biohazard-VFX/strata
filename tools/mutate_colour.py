import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from check_layered_head import compare, runtime_command

ROOT = Path(__file__).resolve().parent.parent
MUTANTS = {
    'wrong-matrix': ('core/colour.c', '.6130974024011883', '.7130974024011883'),
    'encoder-display-skipped': ('core/depth.c', 'out[c][x] = (shoulder(value) - mean[c]) / std[c];', 'out[c][x] = (shoulder(value * 0 + rgb[c] * gain) - mean[c]) / std[c];'),
    'conversion-skipped': ('core/colour.c', 'if (frame->colorspace == STRATA_COLOR_ACESCG)', 'if (1)'),
    'deep-colour-converted': ('backend/runtime.c', 'strata_prediction_to_deep(frame->image,', 'strata_prediction_to_deep(model.image,'),
}


def preload(build, work, source, old, new):
    entries = json.loads((build / 'compile_commands.json').read_text())
    entry = next(item for item in entries if item['file'].endswith('/src/' + source))
    original = Path(entry['file']).read_text()
    assert original.count(old) == 1, (source, old)
    target = work / 'src' / source
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(original.replace(old, new))
    (work / 'src/core').mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'runtime/src/core/internal.h', work / 'src/core/internal.h')
    command = shlex.split(entry['command'])
    command[command.index('-o') + 1] = str(work / 'mutant.o')
    command[command.index(entry['file'])] = str(target)
    subprocess.run(command, cwd=entry['directory'], check=True, capture_output=True, text=True)
    cache = (build / 'CMakeCache.txt').read_text().splitlines()
    ort = next(line.split('=', 1)[1] for line in cache if line.startswith('ONNXRUNTIME_LIBRARY:'))
    library = work / 'mutant.so'
    flags = ['-fsanitize=address,undefined'] if 'STRATA_SANITIZE:BOOL=ON' in cache else []
    subprocess.run([command[0], *flags, '-shared', str(work / 'mutant.o'), '-L' + str(build),
        '-Wl,-rpath,' + str(build), '-Wl,-rpath,' + str(Path(ort).parent), '-lstrata', ort, '-lcrypto', '-ldl', '-lm', '-o', str(library)], check=True)
    if flags:
        asan = subprocess.check_output([command[0], "-print-file-name=libasan.so"], text=True).strip()
        return asan + ":" + str(library)
    return str(library)


def verify(cli, output, case, env):
    command = runtime_command(cli, output, case, 'cpu')
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    assert completed.returncode == 0, ('runtime failed', completed.stderr)
    with np.load(output / (case['id'] + '.npz')) as archive:
        reference = dict(archive)
    return compare(output / (case['id'] + '.deep.exr'), reference, 5e-6, case['colorspace'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    cli = build / 'strata'
    cases = json.loads((ROOT / 'runtime/tests/parity_cases.json').read_text())['layered']
    case = next(item for item in cases if item['colorspace'] == 'lin-rec709')
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        output = work / 'oracle'
        subprocess.run([sys.executable, str(ROOT / 'tools/check_layered_head.py'), '--cli', str(cli),
            '--out', str(output)], check=True, capture_output=True, text=True)
        env = dict(os.environ)
        verify(cli, output, case, env)
        report = [{'name': 'baseline', 'passed': True}]
        for name, (source, old, new) in MUTANTS.items():
            directory = work / name
            directory.mkdir()
            library = preload(build, directory, source, old, new)
            mutant_env = {**env, 'LD_PRELOAD': library}
            try:
                verify(cli, output, case, mutant_env)
            except AssertionError as error:
                message = str(error)
                assert 'Not equal' in message or 'Arrays are not equal' in message, (name, message)
                report.append({'name': name, 'killed': True, 'assertion': message.splitlines()[1]})
            else:
                raise AssertionError('survived: ' + name)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
