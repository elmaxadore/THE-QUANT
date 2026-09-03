#!/usr/bin/env python3
"""End-to-end persistence test: enqueue -> artifacts branch -> pull.

Runs entirely against local bare repos (no GitHub, no network), proving the
queue/artifact branches and the CLI's enqueue/pull round-trip.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from colab import persist  # noqa: E402


class TestPersistRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dq_persist_test_")
        bare = os.path.join(self.tmp, "origin.git")
        subprocess.run(["git", "init", "--bare", "-b", "main", bare],
                       check=True, capture_output=True)
        self.work = os.path.join(self.tmp, "work")
        subprocess.run(["git", "clone", bare, self.work], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", self.work, "config", "user.email",
                        "test@local"], check=True, capture_output=True)
        subprocess.run(["git", "-C", self.work, "config", "user.name",
                        "test"], check=True, capture_output=True)
        with open(os.path.join(self.work, "README.md"), "w") as f:
            f.write("test repo\n")
        subprocess.run(["git", "-C", self.work, "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", self.work, "commit", "-m", "init"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", self.work, "push", "origin", "main"],
                       check=True, capture_output=True)
        os.environ["GITHUB_TOKEN"] = ""  # local paths push without token

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_enqueue_claim_artifact_pull(self):
        # 1. enqueue a job
        persist.enqueue_job(self.work, "20250903-test-dl",
                            {"script": "colab/jobs/download_data.py",
                             "params": {"symbols": "eurusd"},
                             "status": "pending"})
        subprocess.run(["git", "-C", self.work, "fetch", "origin",
                        "colab-jobs"], check=True, capture_output=True)
        spec = json.loads(subprocess.run(
            ["git", "-C", self.work, "show",
             "FETCH_HEAD:20250903-test-dl/job.json"],
            capture_output=True, text=True, check=True).stdout)
        self.assertEqual(spec["status"], "pending")
        self.assertEqual(spec["params"]["symbols"], "eurusd")

        # 2. claim (simulating an agent) -> owner is us
        self.assertTrue(persist.claim_job(
            self.work, "20250903-test-dl", "colab-jobs", "", "worker-a"))
        # a different worker must NOT be able to steal a fresh claim
        self.assertFalse(persist.claim_job(
            self.work, "20250903-test-dl", "colab-jobs", "", "worker-b"))

        # 3. job produces artifacts -> pushed to the artifacts branch
        art = os.path.join(self.tmp, "metrics.json")
        with open(art, "w") as f:
            json.dump({"bars": 123}, f)
        persist.push_artifacts(self.work, [art], subdir="20250903-test-dl",
                               token="")
        subprocess.run(["git", "-C", self.work, "fetch", "origin",
                        "colab-artifacts"], check=True, capture_output=True)
        listed = persist.list_artifacts(self.work)
        self.assertIn("20250903-test-dl/metrics.json", listed)

        # 4. release the job as completed
        persist.release_job(self.work, "20250903-test-dl", "colab-jobs",
                            "", "completed", "rc=0")
        subprocess.run(["git", "-C", self.work, "fetch", "origin",
                        "colab-jobs"], check=True, capture_output=True)
        spec = json.loads(subprocess.run(
            ["git", "-C", self.work, "show",
             "origin/colab-jobs:20250903-test-dl/job.json"],
            capture_output=True, text=True, check=True).stdout)
        self.assertEqual(spec["status"], "completed")


if __name__ == "__main__":
    unittest.main()
