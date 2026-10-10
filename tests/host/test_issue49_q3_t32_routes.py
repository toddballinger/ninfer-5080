#!/usr/bin/env python3
"""Independent boundary fixtures for Issue #49 Q3 dispatch route model."""
import unittest
from tools.analysis.issue49_q3_t32_routes import route, aggregate


class RouteTests(unittest.TestCase):
    def test_special_decode(self):
        self.assertEqual([route(n) for n in range(1, 5)], [
            {"gemv": 1}, {"small2_mma": 1},
            {"small3_pair": 1}, {"small4_mma": 1}
        ])

    def test_precise_spill_reachability(self):
        self.assertNotIn("t32_mma", route(31))
        for n in (32, 33, 63, 64, 128, 255, 256):
            self.assertGreater(route(n)["t32_mma"], 0)

    def test_allow_a8_crossover(self):
        self.assertEqual(route(256), {"t32_mma": 8})
        self.assertEqual(route(257), {"int8": 1})
        self.assertEqual(route(4096), {"int8": 1})
        self.assertEqual(route(4097), {"int8": 2})
        self.assertEqual(route(257, "A16Only")["t32_mma"], 8)

    def test_tails(self):
        self.assertEqual(route(31), {"small16_pair": 1, "small8_pair": 1,
                                     "small4_pair": 1, "small3_pair": 1})
        self.assertEqual(route(35), {"t32_mma": 1, "small3_pair": 1})
        self.assertEqual(route(36), {"t32_mma": 1, "small4_pair": 1})

    def test_histogram_distinguishes_calls_and_launches(self):
        report = aggregate([{"tokens": 32, "calls": 2},
                            {"tokens": 256, "calls": 3},
                            {"tokens": 257, "calls": 5}])
        self.assertEqual(report["operator_calls"], 10)
        self.assertEqual(report["t32_launches"], 2 + 3 * 8)
        self.assertEqual(report["kernel_launch_estimates"]["int8"], 5)
        self.assertIsNone(report["observed_ttft_ms"])

    def test_invalid_cases(self):
        for n in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                route(n)
        with self.assertRaises(ValueError):
            route(4, "unknown")
        with self.assertRaises(ValueError):
            aggregate([{"tokens": 32, "calls": -1}])


if __name__ == "__main__":
    unittest.main()
