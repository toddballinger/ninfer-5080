#!/usr/bin/env python3
"""Issue #45: CPU-only, standard-library sampling oracle; no production imports.

The expected probabilities are algebraic fixtures, NOT outputs copied from
NInfer's CUDA sampler or its existing GPU-backed reference test.
"""
import math
import random
import unittest


def distribution(logits, *, counts=None, presence=0.0, frequency=0.0,
                 temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
                 token_domain=None):
    """Reference policy described in include/ninfer/ops/sampling.h.

    Semantics are intentionally explicit: penalty once, top-k <=20, min-p
    relative to max candidate, then shortest top-p prefix using the weight
    sum BEFORE min-p filtering. This is a contract oracle, not GPU parity.
    """
    n = len(logits) if token_domain is None else token_domain
    if not 1 <= n <= len(logits):
        raise ValueError("invalid token_domain")
    if counts is None:
        counts = [0] * n
    if len(counts) < n:
        raise ValueError("counts too short")
    if temperature <= 0:
        return {min(range(n), key=lambda i: (-logits[i], i)): 1.0}
    if not (0 <= top_p <= 1 and 0 <= min_p <= 1):
        raise ValueError("invalid filter")
    ranked = []
    for i in range(n):
        adjusted = logits[i] - (presence if counts[i] else 0) - frequency * counts[i]
        ranked.append((i, adjusted))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    cap = min(n, top_k if 0 < top_k < 20 else 20)
    ranked = ranked[:cap]
    max_scaled = ranked[0][1] / temperature
    weights = [math.exp(v / temperature - max_scaled) for _, v in ranked]
    prefilter_total = sum(weights)
    min_threshold = min_p * weights[0]
    kept = []
    acc = 0.0
    for (i, _), weight in zip(ranked, weights):
        if weight < min_threshold:
            break
        kept.append((i, weight))
        acc += weight
        if top_p < 1 and acc >= top_p * prefilter_total:
            break
    if not kept:
        kept = [(ranked[0][0], weights[0])]
    z = sum(w for _, w in kept)
    return {i: w / z for i, w in kept}


def seeded_draw(probabilities, seed):
    """Deterministic CPU harness draw; NOT a replica of NInfer counter RNG."""
    u = random.Random(seed).random()
    acc = 0.0
    for token, probability in probabilities.items():
        acc += probability
        if u < acc:
            return token
    return next(reversed(probabilities))


class IndependentAlgebraicFixtures(unittest.TestCase):
    def assert_distribution(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for token, probability in expected.items():
            self.assertAlmostEqual(actual[token], probability, places=12)
        self.assertAlmostEqual(sum(actual.values()), 1, places=12)

    def test_unfiltered_exact_ratios(self):
        # exp(log(6)), exp(log(3)), exp(log(1)) = 6:3:1, exactly 60/30/10%.
        self.assert_distribution(distribution([math.log(6), math.log(3), 0]),
                                 {0: 0.6, 1: 0.3, 2: 0.1})

    def test_penalty_applied_once_not_twice(self):
        # One presence penalty log(2) changes 6:3:1 -> 3:3:1.
        logits = [math.log(6), math.log(3), 0]
        got = distribution(logits, counts=[1, 0, 0], presence=math.log(2))
        self.assert_distribution(got, {0: 3 / 7, 1: 3 / 7, 2: 1 / 7})
        # The incorrect two-pass transform would yield 1.5:3:1 = 3:6:2.
        self.assertGreater(abs(got[0] - 3 / 11), 0.1)

    def test_frequency_and_presence_are_distinct(self):
        # 8:4:2, count=2 on first token, frequency=log(2) -> 2:4:2.
        got = distribution([math.log(8), math.log(4), math.log(2)],
                           counts=[2, 0, 0], frequency=math.log(2))
        self.assert_distribution(got, {0: 0.25, 1: 0.5, 2: 0.25})

    def test_filter_order_discriminator(self):
        # 6:3:1, min_p=.25 drops weight 1. top_p=.65 compares to original
        # total 10: needs weight >=6.5, so keeps 6+3, not just 6.
        # Recomputing top-p denominator after min-p would keep only token 0.
        got = distribution([math.log(6), math.log(3), 0],
                           min_p=0.25, top_p=0.65)
        self.assert_distribution(got, {0: 2 / 3, 1: 1 / 3})

    def test_top_p_shortest_prefix_and_top_k(self):
        logits = [math.log(6), math.log(3), 0]
        self.assert_distribution(distribution(logits, top_p=0.5), {0: 1.0})
        self.assert_distribution(distribution(logits, top_k=2),
                                 {0: 2 / 3, 1: 1 / 3})
        self.assert_distribution(distribution(logits, top_k=1), {0: 1.0})

    def test_temperature_and_greedy_ignore_penalties(self):
        logits = [0.0, 0.0, -3.0]
        self.assert_distribution(distribution(logits, temperature=0,
                                  counts=[10, 0, 0], presence=2, frequency=2),
                                 {0: 1.0})
        self.assert_distribution(distribution([math.log(4), 0], temperature=2),
                                 {0: 2 / 3, 1: 1 / 3})

    def test_token_domain_excludes_padded_rows(self):
        self.assert_distribution(distribution([0, 0, 1000], token_domain=2),
                                 {0: 0.5, 1: 0.5})
        with self.assertRaises(ValueError):
            distribution([0, 1], token_domain=3)

    def test_seeded_harness_replay_not_gpu_equivalence(self):
        p = distribution([math.log(6), math.log(3), 0])
        first = [seeded_draw(p, 42 + i) for i in range(30)]
        self.assertEqual(first, [seeded_draw(p, 42 + i) for i in range(30)])
        self.assertTrue(set(first).issubset(set(p)))


if __name__ == "__main__":
    unittest.main()
