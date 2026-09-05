#!/usr/bin/env python3
"""Tests for the coordinator's chain/retry decisions (enqueue is mocked)."""

import os
import sys
import tempfile
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


class TestRegistry(unittest.TestCase):
    def setUp(self):
        # hermetic: no real reports/, git cat-file always "missing"
        self.tmp = tempfile.mkdtemp(prefix="coord_reg_")
        patcher = mock.patch.object(coordinator, "ROOT", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        git_patcher = mock.patch.object(
            coordinator, "git",
            side_effect=lambda *a, **k: mock.Mock(returncode=1))
        git_patcher.start()
        self.addCleanup(git_patcher.stop)

    def _reg(self, jobs):
        return coordinator.registry(jobs)

    def test_awaiting_data_when_no_jobs(self):
        reg = self._reg({})
        self.assertEqual(reg["summary"]["total"], 6)
        self.assertFalse(reg["summary"]["data_collected"])
        for s in reg["strategies"]:
            self.assertEqual(s["status"], "awaiting-data")
            self.assertEqual(s["progress_pct"], 0)

    def test_queued_train_beats_awaiting_data(self):
        reg = self._reg({"t1": mk_job(TRAIN, "queued")})
        self.assertFalse(reg["summary"]["data_collected"])
        for s in reg["strategies"]:
            self.assertEqual(s["status"], "queued")
            self.assertEqual(s["progress_pct"], 20)
        self.assertEqual(reg["summary"]["active"], 6)

    def test_data_done_progress_and_status(self):
        reg = self._reg({
            "d1": mk_job(DOWNLOAD, "completed"),
            "t1": mk_job(TRAIN, "queued"),
        })
        self.assertTrue(reg["summary"]["data_collected"])
        for s in reg["strategies"]:
            self.assertEqual(s["status"], "queued")
            self.assertEqual(s["progress_pct"], 20)
        self.assertEqual(reg["summary"]["active"], 6)

    def test_claimed_shows_platform_and_progress(self):
        reg = self._reg({
            "d1": mk_job(DOWNLOAD, "completed"),
            "t1": mk_job(TRAIN, "claimed",
                         claimed_by="fv-az123-runners-1"),
        })
        for s in reg["strategies"]:
            self.assertEqual(s["status"], "training")
            self.assertEqual(s["progress_pct"], 85)
            self.assertEqual(s["worked_on_by"], ["github-actions"])

    def test_platform_mapping(self):
        cases = {"abc-runners-9": "github-actions", "colab-x": "colab",
                 "kernel-1": "kaggle", "codespace-xyz": "codespaces"}
        for wid, expected in cases.items():
            reg = self._reg({"t1": mk_job(TRAIN, "claimed",
                                          claimed_by=wid)})
            self.assertEqual(reg["strategies"][0]["worked_on_by"],
                             [expected], wid)


if __name__ == "__main__":
    unittest.main()
