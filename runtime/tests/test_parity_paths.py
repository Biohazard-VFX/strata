import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import parity


class OverflowPaths(unittest.TestCase):
    def test_relative_out_passes_the_written_image(self):
        commands = []

        def run(command, **_):
            commands.append(command)
            return subprocess.CompletedProcess(command, 1, '', 'intercepted')

        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                Path('out').mkdir()
                args = SimpleNamespace(cli=Path('strata'), model=Path('model.onnx'), out=Path('out'),
                                       overflow_field=Path(__file__).parent/'overflow_field.json')
                with mock.patch.object(parity.subprocess, 'run', run), self.assertRaises(AssertionError):
                    parity.check_overflow(args)
                image = Path(commands[0][commands[0].index('--in') + 1])
                self.assertEqual(image, Path('out/fp16_overflow.png'))
                self.assertTrue(image.is_file())
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
