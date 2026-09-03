#!/usr/bin/env python3
"""Tests for the checkpoint module."""

import os
import sys
import tempfile
import shutil
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checkpoint import Checkpoint, TrainingCheckpoint, STAGES, should_resume_job


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dq_ckpt_test_")
        self.ckpt = Checkpoint("test_job", self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initial_progress(self):
        progress = self.ckpt.get_progress()
        self.assertEqual(progress["completed_stages"], 0)
        self.assertEqual(progress["next_stage"], "download")
        self.assertFalse(progress["is_complete"])

    def test_mark_done_and_check(self):
        self.ckpt.mark_done("download", {"symbols": ["eurusd"]})
        self.assertTrue(self.ckpt.is_done("download"))
        self.assertFalse(self.ckpt.is_done("features"))

    def test_save_load_numpy(self):
        data = np.random.randn(100, 12)
        self.ckpt.save("features", data, fmt="numpy")
        loaded = self.ckpt.load("features")
        np.testing.assert_array_equal(data, loaded)

    def test_save_load_dict_numpy(self):
        data = {"X": np.random.randn(100, 12), "y": np.random.randn(100, 1)}
        self.ckpt.save("features", data, fmt="numpy")
        loaded = self.ckpt.load("features")
        np.testing.assert_array_equal(data["X"], loaded["X"])
        np.testing.assert_array_equal(data["y"], loaded["y"])

    def test_save_load_json(self):
        data = {"accuracy": 0.95, "loss": 0.02, "epochs": 50}
        self.ckpt.save("train_mlp", data, fmt="json")
        loaded = self.ckpt.load("train_mlp")
        self.assertEqual(loaded, data)

    def test_progress_after_multiple_stages(self):
        self.ckpt.mark_done("download")
        self.ckpt.mark_done("features")
        progress = self.ckpt.get_progress()
        self.assertEqual(progress["completed_stages"], 2)
        self.assertEqual(progress["next_stage"], "train_mlp")

    def test_all_stages_done(self):
        for stage in STAGES:
            self.ckpt.mark_done(stage)
        progress = self.ckpt.get_progress()
        self.assertTrue(progress["is_complete"])
        self.assertIsNone(progress["next_stage"])

    def test_persistence_across_instances(self):
        self.ckpt.mark_done("download")
        self.ckpt.save("features", {"rows": 5000}, fmt="json")
        ckpt2 = Checkpoint("test_job", self.tmpdir)
        self.assertTrue(ckpt2.is_done("download"))
        loaded = ckpt2.load("features")
        self.assertEqual(loaded["rows"], 5000)


class TestTrainingCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dq_train_ckpt_test_")
        self.ckpt = TrainingCheckpoint("test_job", "train_ckpt", self.tmpdir,
                                       every_n_epochs=10)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_should_save(self):
        self.assertFalse(self.ckpt.should_save(0))
        self.assertFalse(self.ckpt.should_save(5))
        self.assertTrue(self.ckpt.should_save(10))
        self.assertTrue(self.ckpt.should_save(20))

    def test_save_load_latest(self):
        model_state = {"weight": np.random.randn(12, 64), "bias": np.zeros(64)}
        optimizer_state = {"lr": 0.001, "step": 100}
        self.ckpt.save(10, model_state, optimizer_state, 0.05)
        loaded = self.ckpt.load_latest()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["epoch"], 10)
        np.testing.assert_array_equal(loaded["model_state"]["weight"], model_state["weight"])

    def test_resume_epoch(self):
        self.assertEqual(self.ckpt.get_resume_epoch(), 0)
        model_state = {"w": np.zeros(10)}
        self.ckpt.save(20, model_state, {}, 0.03)
        self.assertEqual(self.ckpt.get_resume_epoch(), 20)

    def test_best_loss_tracking(self):
        self.assertEqual(self.ckpt.get_best_loss(), float("inf"))
        self.ckpt.save(10, {"w": np.zeros(5)}, {}, 0.05)
        self.ckpt.save(20, {"w": np.zeros(5)}, {}, 0.03)
        self.assertAlmostEqual(self.ckpt.get_best_loss(), 0.03)

    def test_no_checkpoint_returns_none(self):
        loaded = self.ckpt.load_latest()
        self.assertIsNone(loaded)


class TestShouldResumeJob(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dq_resume_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_checkpoint(self):
        should, path = should_resume_job("nonexistent", self.tmpdir)
        self.assertFalse(should)
        self.assertIsNone(path)

    def test_local_checkpoint_exists(self):
        ckpt = Checkpoint("existing_job", self.tmpdir)
        ckpt.mark_done("download")
        should, path = should_resume_job("existing_job", self.tmpdir)
        self.assertTrue(should)
        self.assertIsNotNone(path)


if __name__ == "__main__":
    unittest.main()
