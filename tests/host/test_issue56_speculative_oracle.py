#!/usr/bin/env python3
"""Independent FP64 oracle for speculative rejection sampling, Issue #56.
Standard library only; no CUDA, NInfer import, or GPU parity claims.
"""
import math
import random
import unittest


def normalize(weights):
    if not weights or any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError("invalid distribution weights")
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("zero distribution mass")
    return tuple(x / total for x in weights)


def residual(p, q):
    if len(p) != len(q):
        raise ValueError("vocabulary mismatch")
    weights = tuple(max(a - b, 0.0) for a, b in zip(p, q))
    return None if math.fsum(weights) == 0.0 else normalize(weights)


def acceptance(p, q, token):
    if q[token] <= 0:
        raise ValueError("cannot propose a zero-q token")
    return min(1.0, p[token] / q[token])


def emitted_distribution(p, q):
    """Exact enumeration of all draft, accept and correction branches."""
    if len(p) != len(q):
        raise ValueError("vocabulary mismatch")
    result = [0.0] * len(p)
    correction = residual(p, q)
    for draft, proposal_mass in enumerate(q):
        if proposal_mass == 0:
            continue
        a = acceptance(p, q, draft)
        result[draft] += proposal_mass * a
        rejected = proposal_mass * (1 - a)
        if rejected:
            if correction is None:
                raise AssertionError("reject without residual mass")
            for token, r in enumerate(correction):
                result[token] += rejected * r
    return tuple(result)


def draw(weights, rng):
    u, cumulative = rng.random(), 0.0
    for token, prob in enumerate(weights):
        cumulative += prob
        if u < cumulative:
            return token
    return len(weights) - 1


def speculative_step(p, q, rng, *, terminal_tokens=frozenset()):
    draft = draw(q, rng)
    if rng.random() < acceptance(p, q, draft):
        emitted, accepted = draft, True
    else:
        correction = residual(p, q)
        if correction is None:
            raise AssertionError("reject without residual mass")
        emitted, accepted = draw(correction, rng), False
    return emitted, accepted, emitted in terminal_tokens


def greedy_step(logits, draft, *, terminal_tokens=frozenset()):
    token = max(range(len(logits)), key=lambda i: (logits[i], -i))
    return token, token == draft, token in terminal_tokens


def speculative_round(target_rows, proposal_rows, seed=1,
                      terminal_tokens=frozenset()):
    """Accepted prefix, then correction/bonus; no tokens past EOS/EOG."""
    if len(target_rows) != len(proposal_rows) + 1:
        raise ValueError("missing bonus target row")
    rng, output, accepted = random.Random(seed), [], 0
    for p, q in zip(target_rows, proposal_rows):
        token, ok, terminal = speculative_step(p, q, rng,
                                               terminal_tokens=terminal_tokens)
        output.append(token)
        if terminal:
            return output, accepted + int(ok), True
        if not ok:
            return output, accepted, False
        accepted += 1
    bonus = draw(target_rows[-1], rng)
    output.append(bonus)
    return output, accepted, bonus in terminal_tokens


class SpeculativeOracleTests(unittest.TestCase):
    def assert_dist(self, actual, expected, tolerance=2e-12):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, delta=tolerance)
        self.assertAlmostEqual(math.fsum(actual), 1.0, places=12)

    def test_exact_two_token_rejection(self):
        p, q = (0.75, 0.25), (0.25, 0.75)
        self.assertEqual(acceptance(p, q, 0), 1.0)
        self.assertAlmostEqual(acceptance(p, q, 1), 1 / 3)
        self.assert_dist(residual(p, q), (1, 0))
        self.assert_dist(emitted_distribution(p, q), p)

    def test_equal_distributions_always_accept(self):
        p = (0.2, 0.3, 0.5)
        self.assertIsNone(residual(p, p))
        self.assert_dist(emitted_distribution(p, p), p)
        self.assertEqual([acceptance(p, p, i) for i in range(3)], [1.] * 3)

    def test_zero_q_and_tiny_q(self):
        p, q = (0.6, 0.4, 0.), (0., 1., 0.)
        self.assert_dist(emitted_distribution(p, q), p)
        with self.assertRaises(ValueError):
            acceptance(p, q, 0)
        self.assert_dist(emitted_distribution(p, (1e-12, 1-1e-12, 0)), p)

    def test_nearly_identical_distributions_keep_nonzero_residual(self):
        # Tiny but nonzero rejection mass must not be rounded to zero
        # by an arbitrary absolute cutoff.
        p = (0.5 + 1e-15, 0.5 - 1e-15)
        q = (0.5, 0.5)
        self.assertIsNotNone(residual(p, q))
        self.assert_dist(residual(p, q), (1.0, 0.0))
        self.assert_dist(emitted_distribution(p, q), p)

    def test_greedy_boundary_and_ties(self):
        self.assertEqual(greedy_step((3, 2), 0), (0, True, False))
        self.assertEqual(greedy_step((3, 2), 1), (0, False, False))
        self.assertEqual(greedy_step((3, 3), 1), (0, False, False))

    def test_eos_acceptance_stops_before_later_tokens(self):
        eos = frozenset({2})
        rows = [(0, 0, 1), (1, 0, 0), (1, 0, 0)]
        q = [(0, 0, 1), (1, 0, 0)]
        self.assertEqual(speculative_round(rows, q, 8, eos), ([2], 1, True))
        rows = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
        q = [(1, 0, 0), (0, 0, 1)]
        self.assertEqual(speculative_round(rows, q, 8, eos), ([0, 2], 2, True))

    def test_bonus_and_seed_replay(self):
        rows = [(1, 0), (0, 1), (0.4, 0.6)]
        q = [(1, 0), (0, 1)]
        a = speculative_round(rows, q, 123)
        self.assertEqual(a, speculative_round(rows, q, 123))
        self.assertEqual(a[0][:2], [0, 1])
        self.assertEqual((a[1], len(a[0])), (2, 3))

    def test_randomized_exact_marginals(self):
        rng = random.Random(5600)
        for _ in range(500):
            n = rng.randrange(2, 12)
            p = normalize([rng.random() + 1e-6 for _ in range(n)])
            q = normalize([rng.random() + 1e-6] +
                          [rng.random() if rng.random() > .2 else 0
                           for _ in range(n - 1)])
            self.assert_dist(emitted_distribution(p, q), p, 3e-12)

    def test_invalid_distributions(self):
        for weights in ((), (0, 0), (-1, 2), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                normalize(weights)
        with self.assertRaises(ValueError):
            emitted_distribution((1,), (0.5, 0.5))
        with self.assertRaises(ValueError):
            speculative_round([(1, 0)], [(1, 0)])


if __name__ == "__main__":
    unittest.main()
