#!/usr/bin/env python3
"""Synthetic regression coverage for read-only Issue #58 analyzer."""
import json
import tempfile
import unittest
from pathlib import Path
from tools.bench.issue58_admission_forensics import analyse, quantile

class ForensicsTests(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(quantile([1,2,3],.5),2)
        self.assertEqual(quantile([1,3],.5),2)
        self.assertIsNone(quantile([],.95))
    def test_completed_only_and_occupancy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"server").mkdir()
            (root/"forensics").mkdir()
            events=[
                {"event":"request_start","request":{"request_id":1}},
                {"event":"request_start","request":{"request_id":2}},
                {"event":"request_done","request":{"request_id":1},"result":{"completion_tokens":7}},
                {"event":"request_error","request":{"request_id":2},"error":{"message":"inference request expired while waiting for admission"}},
            ]
            (root/"server"/"test.jsonl").write_text("\n".join(map(json.dumps,events))+"\n")
            (root/"forensics"/"test.stderr.log").write_text(
                "[info] ninfer-serve: [req 1] done finish=stop_token ttft=1500ms wall=2s\n"
                "[info] ninfer-serve: [req 2] done finish=stop_token ttft=900000ms wall=900s\n"
                "[info] ninfer-serve: throughput interval=1.000s prefill=0 decode=2 running=1 waiting=1 avg_decode_batch=1\n")
            result=analyse(root)
            self.assertEqual(result["request_done_events"],1)
            self.assertEqual(result["queue_timeout_events"],1)
            self.assertEqual(result["ttft_samples"],1)
            self.assertEqual(result["ttft_seconds"]["max"],1.5)
            self.assertEqual(result["running_waiting_seconds"]["1:1"],1.0)
    def test_invalid_json_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"server").mkdir()
            (root/"forensics").mkdir()
            (root/"server"/"bad.jsonl").write_text("{bad json}")
            (root/"forensics"/"test.stderr.log").write_text("")
            with self.assertRaises(ValueError):
                analyse(root)

if __name__=="__main__":
    unittest.main()
