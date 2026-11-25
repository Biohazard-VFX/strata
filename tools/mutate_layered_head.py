import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
MUTANTS = {
    'unpremultiplied-colour': ('image->rgba[i * 4 + c] * a', 'image->rgba[i * 4 + c]'),
    'wrong-depth-units': ('1 / inv : INFINITY', '2 / inv : INFINITY'),
    'retain-empty-layers': ('if (a == 0 || remaining == 0)', 'if (0)'),
    'merge-equal-depths': ('previous = z;', 'if (z == previous && a > 0)\n\t\t\t\tcontinue;\n\t\t\tprevious = z;'),
    'incomplete-coverage': ('if (remaining != 0)', 'if (0)'),
    'reverse-depth-order': (' || z < previous', ' || (z < previous && 0)'),
}


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
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
