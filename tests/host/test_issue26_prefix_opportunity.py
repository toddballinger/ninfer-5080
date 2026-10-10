#!/usr/bin/env python3
"""Issue #26 host-only fixture tests; no NInfer or CUDA imports."""
import unittest
from tools.analysis.issue26_prefix_opportunity import (
    Span, prompt, lcp, evaluate, campaign, summary)


class PrefixOpportunityTests(unittest.TestCase):
    def test_exact_identical_and_append(self):
        a = prompt(Span("a", 10))
        b = prompt(Span("a", 10), Span("b", 3))
        self.assertEqual(lcp(a, a), 10)
        self.assertEqual(lcp(a, b), 10)
        self.assertEqual(lcp(b, a), 10)

    def test_changed_content_not_reusable(self):
        a = prompt(Span("a", 10), Span("x", 5))
        b = prompt(Span("a", 10), Span("y", 5))
        self.assertEqual(lcp(a, b), 10)

    def test_vision_digest_is_identity(self):
        a, b = campaign()["vision_identity_changed"]
        self.assertEqual(lcp(a, b), 1024)
        c, d = campaign()["vision_identity_same"]
        self.assertEqual(lcp(c, d), len(c))

    def test_position_change_is_identity(self):
        a, b = campaign()["position_identity_changed"]
        self.assertEqual(lcp(a, b), 1024)

    def test_alternating_conversation_checkpoint_opportunity(self):
        events = evaluate(campaign()["alternating_conversations"], 6)
        self.assertGreater(events[2]["additional_reusable_tokens_upper_bound"], 0)
        self.assertGreater(events[3]["additional_reusable_tokens_upper_bound"], 0)
        self.assertTrue(all(e["bounded_snapshot_lcp_upper_bound"] >=
                            e["resident_lcp_upper_bound"] for e in events))

    def test_zero_checkpoints_equal_resident(self):
        events = evaluate(campaign()["alternating_conversations"], 0)
        self.assertTrue(all(e["additional_reusable_tokens_upper_bound"] == 0
                            for e in events))

    def test_long_context_has_small_unmatched_append(self):
        events = evaluate(campaign()["long_118k_tiny_suffix"])
        self.assertEqual(events[1]["prompt_tokens"], 118816)
        self.assertEqual(events[1]["resident_lcp_upper_bound"], 118784)
        self.assertEqual(events[1]["resident_unmatched_tokens"], 32)

    def test_synthetic_metadata_and_bounds(self):
        report = summary()
        self.assertTrue(report["synthetic"])
        self.assertFalse(report["measured_runtime"])
        self.assertGreaterEqual(len(report["scenario_results"]), 9)
        for record in report["scenario_results"].values():
            self.assertLessEqual(record["snapshot_unmatched_tokens"],
                                 record["resident_unmatched_tokens"])
        with self.assertRaises(ValueError):
            evaluate([], -1)
        with self.assertRaises(ValueError):
            Span("bad", 0)


if __name__ == "__main__":
    unittest.main()
