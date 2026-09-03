#!/usr/bin/env python3
"""Unit tests for THE-QUANT Google Colab Training Bridge."""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from colab.colab_bridge import ColabClient, save_colab_config, load_colab_config
from colab.colab_worker import ColabWorkerHandler, detect_hardware_runtime, SERVER_TOKEN


class TestColabBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import colab.colab_worker
        cls.port = 8099
        cls.token = "test-secret-token"
        colab.colab_worker.SERVER_TOKEN = cls.token
        cls.server = ThreadingHTTPServer(("127.0.0.1", cls.port), ColabWorkerHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.url = f"http://127.0.0.1:{cls.port}"
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_hardware_detection(self):
        hw = detect_hardware_runtime()
        self.assertIn("accelerator_type", hw)
        self.assertIn("cpu_count", hw)

    def test_client_health_check(self):
        client = ColabClient(self.url, self.token)
        self.assertTrue(client.health_check())

    def test_unauthorized_access(self):
        client = ColabClient(self.url, "wrong-token")
        self.assertFalse(client.health_check())

    def test_get_runtime_info(self):
        client = ColabClient(self.url, self.token)
        runtime = client.get_runtime_info()
        self.assertIn("accelerator_type", runtime)

    def test_training_job_execution_and_artifact_collection(self):
        client = ColabClient(self.url, self.token)
        job_id = client.submit_training_job(
            model_type="gbdt",
            parameters={"n_samples": 500, "n_estimators": 5}
        )
        self.assertTrue(job_id.startswith("job-"))

        jinfo = client.wait_for_job(job_id, poll_interval=0.2, verbose=False)
        self.assertEqual(jinfo["status"], "completed", f"Job failed: stdout={jinfo.get('stdout_tail')}, stderr={jinfo.get('stderr_tail')}")
        self.assertEqual(jinfo["returncode"], 0)

        # Test collecting artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            collected = client.collect_all_artifacts(job_id, output_dir=tmpdir)
            self.assertGreaterEqual(len(collected), 1)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "latest.onnx")))


if __name__ == "__main__":
    unittest.main()
