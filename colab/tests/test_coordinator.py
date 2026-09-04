#!/usr/bin/env python3
"""Tests for the coordinator's chain/retry decisions (enqueue is mocked)."""

import os
import sys
import time
import unittest
from unittest import mock

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
sys.path.insert(0, os.path.join(REPO, "deploy"))
import coordinator  # noqa: E402

DOWNLOAD = coordinator.DOWNLOAD_JOB
TRAIN = coordinator.TRAIN_JOB


def mk_job(script, status, **extra):
    spec = {"script": script, "status": status,
            "params": {"symbols": "eurusd,gbpusd,xauusd",
                       "start": "2025-01-01"}}
    spec.update(extra)
    return spec


class TestChain(unittest.TestCase):
    def setUp(self):
        self.enqueued = []
        patcher = mock.patch.object(coordinator, "enqueue",
                                    side_effect=lambda n, s:
                                    self.enqueued.append((n, s)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_data_done_chains_training_with_same_symbols(self):
        jobs = {"d1": mk_job(DOWNLOAD, "completed",
                             params={"symbols": "usdjpy,usdcad",
                                     "start": "2024-06-01"})}
        changed = coordinator.chain(jobs, {"attempts": {}})
        self.assertTrue(changed)
        self.assertEqual(len(self.enqueued), 1)
        name, spec = self.enqueued[0]
        self.assertEqual(spec["script"], TRAIN)
        self.assertEqual(spec["params"]["train_args"],
                         ["--symbols", "usdjpy,usdcad", "--start",
                          "2024-06-01"])

    def test_no_double_chain_when_train_already_queued(self):
        jobs = {"d1": mk_job(DOWNLOAD, "completed"),
                "t1": mk_job(TRAIN, "queued")}
        self.assertFalse(coordinator.chain(jobs, {"attempts": {}}))
        self.assertEqual(self.enqueued, [])

    def test_no_chain_when_data_not_completed(self):
        jobs = {"d1": mk_job(DOWNLOAD, "claimed")}
        self.assertFalse(coordinator.chain(jobs, {"attempts": {}}))
        self.assertEqual(self.enqueued, [])

    def test_failed_job_requeued_after_cooldown(self):
        jobs = {"d1": mk_job(DOWNLOAD, "failed",
                             claimed_at=time.time() - 7201)}
        st = {"attempts": {}}
        self.assertTrue(coordinator.chain(jobs, st))
        name, spec = self.enqueued[0]
        self.assertTrue(name.endswith("-retry1"))
        self.assertEqual(spec["status"], "queued")
        self.assertNotIn("claimed_by", spec)
        self.assertEqual(st["attempts"]["d1"], 1)

    def test_no_retry_within_cooldown(self):
        jobs = {"d1": mk_job(DOWNLOAD, "failed",
                             claimed_at=time.time() - 60)}
        self.assertFalse(coordinator.chain(jobs, {"attempts": {}}))
        self.assertEqual(self.enqueued, [])

    def test_no_retry_after_max_attempts(self):
        jobs = {"d1": mk_job(DOWNLOAD, "failed",
                             claimed_at=time.time() - 7201)}
        st = {"attempts": {"d1": coordinator.MAX_ATTEMPTS}}
        self.assertFalse(coordinator.chain(jobs, st))
        self.assertEqual(self.enqueued, [])


if __name__ == "__main__":
    unittest.main()
