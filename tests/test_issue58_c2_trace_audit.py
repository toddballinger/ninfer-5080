import tempfile
import unittest
from pathlib import Path
from tools.bench.issue58_c2_trace_audit import audit

class TestIssue58C2TraceAudit(unittest.TestCase):
    def capture(self, content):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stderr.log"
            path.write_text(content, encoding="utf-8")
            return audit([path])

    def test_empty(self):
        r = self.capture("")
        self.assertEqual(r["selected_admission_queue_ms"]["n"], 0)
        self.assertIsNone(r["selected_admission_queue_ms"]["p95"])

    def test_selected_admission_only(self):
        r = self.capture("[ADMISSION-QUEUE-TIMING] id=7 lane=0 queue_wait_ms=125 event=selected_for_admission\n")
        self.assertEqual(r["selected_admission_queue_ms"]["p95"], 125)
        self.assertEqual(r["selected_admissions"][0]["request_id"], 7)
        self.assertFalse(r["unrecognised_or_invalid_lines"])

    def test_multi_trace_with_no_causal_join(self):
        data = "\n".join((
            "[ADMISSION-TRACE] head=7 queued=2 active=1 reason=vacant_lane_not_admittable "
            "head_pages_main=2 head_pages_backend=1 used_pages_main=10 used_pages_backend=3 used_lanes=1 "
            "capacity_pages_main=18 capacity_pages_backend=8 capacity_lanes=2 "
            "deadline_remaining_ms=600000 protection_epoch=0 protection_phase=none",
            "[ADMISSION-LANE] head=7 lane=0 result=occupied",
            "[ADMISSION-LANE] head=7 lane=1 result=not_admittable",
            "[ADMISSION-KV-DENIAL] lane=1 path=direct pool=main cause=insufficient_physical_pages old=2 reclaimable=0 requested=12 entitled=10 logical=100 physical=18",
            "[ADMISSION-QUEUE-TIMING] id=7 lane=1 queue_wait_ms=100 event=selected_for_admission",
            "[ADMISSION-QUEUE-TIMING] id=8 lane=0 queue_wait_ms=300 event=selected_for_admission",
        )) + "\n"
        r = self.capture(data)
        self.assertEqual(r["selected_admission_queue_ms"]["p50"], 200)
        self.assertEqual(r["selected_admission_queue_ms"]["p95"], 290)
        self.assertEqual(r["counts"]["kv_insufficient_physical_pages"], 1)
        self.assertEqual(len(r["selected_admissions"]), 2)
        self.assertNotIn("cause", r["selected_admissions"][0])
        self.assertFalse(r["unrecognised_or_invalid_lines"])

    def test_truncated_summary_rejected(self):
        r = self.capture(
            "[ADMISSION-TRACE] head=7 queued=2 active=1 reason=vacant_lane_not_admittable "
            "head_pages_main=2\\n"
        )
        self.assertEqual(r["counts"].get("summary", 0), 0)
        self.assertEqual(len(r["unrecognised_or_invalid_lines"]), 1)

    def test_summary_trailing_data_rejected(self):
        r = self.capture(
            "[ADMISSION-TRACE] head=7 queued=2 active=1 reason=no_vacant_lane "
            "head_pages_main=2 head_pages_backend=1 used_pages_main=10 used_pages_backend=3 used_lanes=1 "
            "capacity_pages_main=18 capacity_pages_backend=8 capacity_lanes=2 "
            "deadline_remaining_ms=600000 protection_epoch=0 protection_phase=none extra=1\\n"
        )
        self.assertEqual(r["counts"].get("summary", 0), 0)
        self.assertEqual(len(r["unrecognised_or_invalid_lines"]), 1)

    def test_reject_malformed(self):
        r = self.capture("[ADMISSION-QUEUE-TIMING] id=1 lane=0 queue_wait_ms=-1 event=selected_for_admission\n")
        self.assertEqual(len(r["unrecognised_or_invalid_lines"]), 1)

    def test_no_assumed_journal_prefix(self):
        r = self.capture("Oct 10 host [ADMISSION-QUEUE-TIMING] id=1 lane=0 queue_wait_ms=42 event=selected_for_admission\n")
        self.assertEqual(r["selected_admission_queue_ms"]["n"], 0)

    def test_sampling_and_policy(self):
        r = self.capture("[ADMISSION-DEFERRAL] head=3 queued=2 reason=short_lane_policy_hold\n")
        self.assertEqual(r["counts"]["deferral_short_lane_policy_hold"], 1)

if __name__ == "__main__":
    unittest.main()
