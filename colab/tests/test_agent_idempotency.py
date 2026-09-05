#!/usr/bin/env python3
"""Re-run safety tests for the remote agent (the "run everything again"
guarantee): repo detection, repair of broken clones, and agent take-over."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import agent  # noqa: E402


class TestRerunSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dq_agent_test_")
        # local "origin" so clone tests need no network
        self.origin = os.path.join(self.tmp, "origin.git")
        seed = os.path.join(self.tmp, "seed")
        subprocess.run(["git", "init", "-b", "main", seed], check=True,
                       capture_output=True)
        for k, v in [("user.email", "t@l"), ("user.name", "t")]:
            subprocess.run(["git", "-C", seed, "config", k, v], check=True,
                           capture_output=True)
        os.makedirs(os.path.join(seed, "colab"))
        with open(os.path.join(seed, "colab", "agent.py"), "w") as f:
            f.write("# agent\n")
        with open(os.path.join(seed, "README.md"), "w") as f:
            f.write("x\n")
        subprocess.run(["git", "-C", seed, "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", seed, "commit", "-m", "init"],
                       check=True, capture_output=True)
        subprocess.run(["git", "clone", "--bare", seed, self.origin],
                       check=True, capture_output=True)
        self.dest = os.path.join(self.tmp, "repo")
        self._orig_url = agent.REPO_URL
        agent.REPO_URL = self.origin

    def tearDown(self):
        agent.REPO_URL = self._orig_url
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_is_repo_dir_and_file(self):
        self.assertFalse(agent.is_repo(self.tmp))
        subprocess.run(["git", "init", "-q", os.path.join(self.tmp, "r1")],
                       check=True, capture_output=True)
        self.assertTrue(agent.is_repo(os.path.join(self.tmp, "r1")))
        # worktree-style .git file pointer also counts
        wf = os.path.join(self.tmp, "r2")
        os.makedirs(wf)
        with open(os.path.join(wf, ".git"), "w") as f:
            f.write("gitdir: somewhere\n")
        self.assertTrue(agent.is_repo(wf))

    def test_repair_or_clone_first_run(self):
        out = agent.repair_or_clone(self.dest)
        self.assertEqual(out, self.dest)
        self.assertTrue(agent.is_repo(self.dest))

    def test_repair_or_clone_second_run_is_noop(self):
        agent.repair_or_clone(self.dest)
        out = agent.repair_or_clone(self.dest)  # re-run must not crash/clone
        self.assertEqual(out, self.dest)
        self.assertTrue(agent.is_repo(self.dest))
        # no .broken- leftovers on the happy path
        self.assertEqual([p for p in os.listdir(self.tmp)
                          if ".broken-" in p], [])

    def test_repair_or_clone_heals_broken_checkout(self):
        agent.repair_or_clone(self.dest)
        # simulate an interrupted/corrupted clone
        subprocess.run(["rm", "-rf", os.path.join(self.dest, ".git")],
                       check=True)
        out = agent.repair_or_clone(self.dest)
        self.assertTrue(agent.is_repo(out))
        self.assertEqual(len([p for p in os.listdir(self.tmp)
                              if ".broken-" in p]), 1)

    def test_take_over_stale_pidfile(self):
        # stale pid pointing at a non-agent/nonexistent process
        with open(agent.PIDFILE, "w") as f:
            f.write("999999999")
        self.assertFalse(agent.take_over_previous_agent())
        self.assertFalse(os.path.exists(agent.PIDFILE))


if __name__ == "__main__":
    unittest.main()
