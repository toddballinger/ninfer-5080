#!/usr/bin/env python3
"""Issue #58 offline event timeline regression fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from tools.bench.issue58_event_timeline import inspect_events, read_jsonl


def event(kind, rid, ts, *, instance="alpha"):
    return {"event": kind, "request": {"request_id": rid},
            "timestamp_unix_ms": ts, "server_instance_id": instance}


class TimelineTests(unittest.TestCase):
    def inspect(self, rows):
        return inspect_events([("sample.jsonl", i + 1, row)
                               for i, row in enumerate(rows)])

    def test_start_done_elapsed_not_queue_wait(self):
        result = self.inspect([event("request_start", 1, 100),
                               event("request_done", 1, 600)])
        self.assertEqual(result["records"][0]["start_to_terminal_ms"], 500)
        self.assertIsNone(result["records"][0]["queue_wait_ms"])
        self.assertFalse(result["admission_directly_observed"])

    def test_error_and_rejected_are_terminal(self):
        report = self.inspect([event("request_start", 1, 1),
                               event("request_error", 1, 5),
                               event("request_rejected", 2, 8)])
        self.assertEqual(report["request_count"], 2)
        self.assertEqual(report["records"][0]["terminal_kind"], "request_error")
        self.assertIn("missing_start", report["records"][1]["anomalies"])

    def test_duplicate_and_out_of_order(self):
        report = self.inspect([event("request_done", 1, 10),
                               event("request_start", 1, 20),
                               event("request_start", 1, 22)])
        self.assertIn("duplicate_start", report["records"][0]["anomalies"])
        self.assertIn("terminal_before_start", report["records"][0]["anomalies"])
        self.assertIsNone(report["records"][0]["start_to_terminal_ms"])

    def test_multiple_terminal(self):
        report = self.inspect([event("request_start", 1, 1),
                               event("request_done", 1, 10),
                               event("request_error", 1, 11)])
        self.assertIn("multiple_terminal", report["records"][0]["anomalies"])
        self.assertIsNone(report["records"][0]["start_to_terminal_ms"])

    def test_server_namespaces_do_not_conflate_ids(self):
        report = self.inspect([event("request_start", 1, 1, instance="a"),
                               event("request_done", 1, 2, instance="a"),
                               event("request_start", 1, 3, instance="b"),
                               event("request_done", 1, 4, instance="b")])
        self.assertEqual(report["request_count"], 2)

    def test_missing_instance_separates_files(self):
        rows = [("one", 1, {**event("request_start", 1, 1), "server_instance_id": None}),
                ("two", 1, {**event("request_done", 1, 3), "server_instance_id": None})]
        self.assertEqual(inspect_events(rows)["request_count"], 2)

    def test_invalid_or_missing_timestamps_are_unusable(self):
        rows = [event("request_start", 1, True),
                event("request_start", 2, -1),
                event("request_start", 3, 5)]
        result = self.inspect(rows)
        self.assertEqual(result["unusable_events"], 2)
        self.assertEqual(result["request_count"], 1)

    def test_file_io_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "capture.jsonl"
            p.write_text("\n".join(map(json.dumps,
                [event("request_start", 9, 100),
                 event("request_done", 9, 200)])) + "\n")
            self.assertEqual(inspect_events(read_jsonl([p]))["request_count"], 1)

    def test_malformed_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.jsonl"
            p.write_text("{oops}\n")
            with self.assertRaisesRegex(ValueError, "malformed JSONL"):
                inspect_events(read_jsonl([p]))


if __name__ == "__main__":
    unittest.main()
