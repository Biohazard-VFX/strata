import argparse
import json
import os
import sys

from pathlib import Path
import subprocess
import tempfile

from mutate_colour import preload, verify

ROOT = Path(__file__).resolve().parent.parent
MUTANTS = {
    'unpremultiplied-colour': ('image->rgba[i * 4 + c] * a', 'image->rgba[i * 4 + c]'),
    'wrong-depth-units': ('1 / inv : INFINITY', '2 / inv : INFINITY'),
    'retain-empty-layers': ('if (a == 0 || remaining == 0)', 'if (0)'),
    'merge-equal-depths': ('previous = z;', 'if (z == previous && a > 0)\n\t\t\t\tcontinue;\n\t\t\tprevious = z;'),
    'incomplete-coverage': ('if (remaining != 0)', 'if (0)'),
    'reverse-depth-order': (' || z < previous', ' || (z < previous && 0)'),
}


TILE_MUTANTS = {
    'halo-x0': ('x > 32 ? x - 32 : 0', 'x > 16 ? x - 16 : 0'),
    'halo-y0': ('y > 32 ? y - 32 : 0', 'y > 16 ? y - 16 : 0'),
    'halo-x1': ('x + 288 < width ? x + 288 : width', 'x + 272 < width ? x + 272 : width'),
    'halo-y1': ('y + 288 < height ? y + 288 : height', 'y + 272 < height ? y + 272 : height'),
    'core-offset': ('(size_t)(x - tile->x0)];', '(size_t)(x - tile->x0 + (x == tile->x && x > 0))];'),
    'feature-stride': ('features[c * padded + padded_pixel]', 'features[c * (padded - 1) + padded_pixel]'),
}


def tile_mutations(build, work):
    output = work / 'oracle'
    subprocess.run([sys.executable, str(ROOT / 'tools/check_layered_head.py'),
        '--cli', str(build / 'strata'), '--out', str(output)], check=True, capture_output=True, text=True)
    cases = json.loads((ROOT / 'runtime/tests/parity_cases.json').read_text())['layered']
    case = next(item for item in cases if item['id'] == 'layered_tiles')
    env = dict(os.environ)
    verify(build / 'strata', output, case, env)
    report = [{'name': 'tile-baseline', 'passed': True}]
    for name, (old, new) in TILE_MUTANTS.items():
        directory = work / name
        directory.mkdir()
        library = preload(build, directory, 'backend/runtime.c', old, new)
        try:
            verify(build / 'strata', output, case, {**env, 'LD_PRELOAD': library})
        except AssertionError as error:
            assert 'Not equal' in str(error), (name, str(error))
            report.append({'name': name, 'killed': True, 'assertion': str(error).splitlines()[1]})
        else:
            raise AssertionError('survived: ' + name)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    source = (ROOT / 'runtime/src/core/deep.c').read_text()
    report = []
    cache = (args.build / "CMakeCache.txt").read_text()
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if "STRATA_SANITIZE:BOOL=ON" in cache else []
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        for name, mutation in [('baseline', None), *MUTANTS.items()]:
            changed = source
            if mutation is not None:
                old, new = mutation
                assert source.count(old) == 1, (name, source.count(old))
                changed = source.replace(old, new)
            path = work / (name + '.c')
            path.write_text(changed)
            executable = work / name
            subprocess.run(['cc', *flags, '-std=c11', '-Wall', '-Wextra', '-Werror',
                '-I' + str(ROOT / 'runtime/src/core'), '-I' + str(ROOT / 'runtime/include'),
                '-I' + str(ROOT / 'contract'),
                str(ROOT / 'runtime/tests/deep.c'), str(path), '-L' + str(args.build.resolve()),
                '-Wl,-rpath,' + str(args.build.resolve()), '-lstrata', '-lm', '-o', str(executable)],
                check=True, text=True)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            assert (result.returncode == 0) == (mutation is None), (name, result)
            assert mutation is None or ': ' in result.stderr, (name, result)
            report.append({'name': name, 'exit_code': result.returncode})
        report.extend(tile_mutations(args.build.resolve(), work))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
