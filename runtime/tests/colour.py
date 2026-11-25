import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main():
    fixture, cli = map(Path, sys.argv[1:])
    reference = json.loads((ROOT / 'runtime/tests/colour_reference.json').read_text())
    plate = np.array([[1, 0, 0, 1], [0, 1, 0, .5], [0, 0, 1, .25], [-1, 2, 4, 1]], np.float32)
    ap1 = np.asarray(reference['to_ap0']['acescg'])
    for name, to_ap0 in reference['to_ap0'].items():
        matrix = np.eye(3) if name == 'acescg' else np.linalg.solve(ap1, np.asarray(to_ap0))
        result = subprocess.run([str(fixture), name], capture_output=True, check=True)
        converted = np.frombuffer(result.stdout, np.float32).reshape(4, 4)
        np.testing.assert_allclose(converted[:3, :3].T, matrix, atol=1e-6, rtol=0)
        expected = (plate[:, :3].astype(np.float64) @ matrix.T).astype(np.float32)
        np.testing.assert_array_equal(converted[:, :3], expected)
        np.testing.assert_array_equal(converted[:, 3], plate[:, 3])
        np.testing.assert_allclose(converted[:, :3] @ np.linalg.inv(matrix).T, plate[:, :3], atol=1e-6, rtol=0)
    command = [str(cli), 'depth', '--model', 'missing/model.onnx', '--in', 'plate.exr', '--out-depth', 'depth.exr']
    for flags in ([], ['--in-colorspace', 'LogC'], ['--in-colorspace', 'S-Log3'], ['--in-colorspace', 'srgb']):
        result = subprocess.run(command + flags, capture_output=True, text=True)
        assert result.returncode == 2, result
        assert '--in-colorspace acescg|aces2065-1|lin-rec709|lin-rec2020' in result.stderr, result
    print('colour reference matrices, exact model RGB, round-trip, alpha and CLI rejection passed')


if __name__ == '__main__':
    main()
