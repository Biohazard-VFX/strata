import json
from pathlib import Path
import unittest

import numpy as np

from parity import CASE_KEYS, LIMITS, gate, statistics

SHAPE = (100, 100)


def truth():
    y, x = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(np.float64)
    return .5 + .25*np.sin(x/17) + .125*np.cos(y/11)


def alternating(amplitude):
    signs = np.where(np.indices(SHAPE).sum(0) % 2, 1., -1.)
    return 1 + amplitude*signs


def fraction(share, amplitude):
    scale = np.ones(SHAPE)
    scale.flat[:int(round(share*scale.size))] += amplitude
    return scale


def failures(native, oracle, fp32):
    native, oracle, fp32 = (np.asarray(value, np.float32) for value in (native, oracle, fp32))
    return gate(statistics(native, oracle)[0], statistics(native, fp32)[0], statistics(oracle, fp32)[0])


class ParityGates(unittest.TestCase):
    def test_margin_is_tighter_than_the_matching_caps(self):
        self.assertLess(LIMITS['fp32_margin'], min(LIMITS['median'], LIMITS['signed_bias']))

    def setUp(self):
        self.fp32 = truth()
        self.oracle = self.fp32*alternating(.3*LIMITS['median'])

    def test_accepts_native_equal_to_oracle(self):
        self.assertEqual(failures(self.oracle, self.oracle, self.fp32), [])

    def test_accepts_noise_within_every_gate(self):
        noise = alternating(.4*LIMITS['median'])*fraction(5e-4, .5*LIMITS['p999'])
        self.assertEqual(failures(self.fp32*noise, self.oracle, self.fp32), [])

    def test_accepts_native_as_far_from_fp32_as_the_oracle(self):
        native = self.fp32*(1 - .4*LIMITS['signed_bias'])
        oracle = self.fp32*(1 + .4*LIMITS['signed_bias'])
        self.assertEqual(failures(native, oracle, self.fp32), [])

    def test_uniform_scale_fails_every_bias_and_median_gate(self):
        self.assertEqual(failures(self.oracle*1.009, self.oracle, self.fp32),
                         ['median', 'signed_bias', 'fp32_median', 'fp32_signed_bias'])

    def test_rejects_interior_spike(self):
        native = self.oracle.copy()
        native[50, 50] *= 1 + 1.2*LIMITS['interior_max']
        self.assertEqual(failures(native, self.oracle, self.fp32), ['interior_max'])

    def test_rejects_p999_tail(self):
        native = self.oracle*fraction(2e-3, 1.5*LIMITS['p999'])
        self.assertEqual(failures(native, self.oracle, self.fp32), ['p999'])

    def test_rejects_median_against_matching_oracle(self):
        oracle = self.fp32*alternating(1.5*LIMITS['median'])
        self.assertEqual(failures(self.fp32, oracle, self.fp32), ['median'])

    def test_rejects_signed_bias_against_matching_oracle(self):
        oracle = self.fp32/fraction(.4, 3*LIMITS['signed_bias'])
        self.assertEqual(failures(self.fp32, oracle, self.fp32), ['signed_bias'])

    def test_rejects_median_worse_than_oracle_against_fp32(self):
        native = self.fp32*alternating((LIMITS['fp32_margin'] + LIMITS['median'])/2)
        self.assertEqual(failures(native, self.fp32, self.fp32), ['fp32_median'])

    def test_rejects_signed_bias_worse_than_oracle_against_fp32(self):
        shift = (LIMITS['fp32_margin'] + LIMITS['signed_bias'])/2
        oracle = self.fp32*alternating(shift)
        native = self.fp32*(1 + shift)
        self.assertEqual(failures(native, oracle, self.fp32), ['fp32_signed_bias'])

    def test_rejects_nonfinite_statistics(self):
        stats = statistics(self.oracle.astype(np.float32), self.oracle.astype(np.float32))[0]
        stats['median_relative'] = float('nan')
        self.assertIn('median', gate(stats, stats, stats))

    def test_cases_cannot_exempt_gates(self):
        cases = json.loads((Path(__file__).parent/'parity_cases.json').read_text())['depth']
        for case in cases:
            self.assertLessEqual(set(case), CASE_KEYS, case['id'])


if __name__ == '__main__':
    unittest.main()
