#!/usr/bin/env python3
"""THE QUANT — Python Client SDK for Google Colab Remote Training.

Provides the `ColabClient` class to seamlessly connect to a Colab runtime,
query available hardware (GPU/TPU/CPU), submit training scripts, stream live logs,
and collect model artifacts (.onnx files) directly into `THE-QUANT/models/`.
"""

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "state", "colab_config.json")


def save_colab_config(url: str, token: str):
    """Saves the Colab connection details to state/colab_config.json."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    cfg = {
        "url": url.rstrip("/"),
        "token": token,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[Colab SDK] Saved Colab connection profile -> {CONFIG_PATH}")


def load_colab_config() -> Optional[Dict[str, str]]:
    """Loads saved Colab connection settings from state/colab_config.json or environment."""
    env_url = os.environ.get("COLAB_URL")
    env_token = os.environ.get("COLAB_TOKEN")
    if env_url and env_token:
        return {"url": env_url.rstrip("/"), "token": env_token}

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                if data.get("url") and data.get("token"):
                    return data
        except Exception:
            pass
    return None


class ColabClient:
    """Client for controlling and retrieving training outputs from Colab worker."""

    def __init__(self, url: str, token: str, timeout: int = 30):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Colab-Token": self.token,
            "Content-Type": "application/json",
            "User-Agent": "THE-QUANT-ColabClient/4.0"
        }

    def _request(self, endpoint: str, method: str = "GET", data: Optional[dict] = None) -> Tuple[int, dict]:
        target_url = f"{self.url}{endpoint}"
        payload = json.dumps(data).encode("utf-8") if data is not None else None

        req = urllib.request.Request(target_url, data=payload, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status_code = resp.status
                body = resp.read().decode("utf-8")
                return status_code, json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if e.fp else "{}"
            try:
                err_json = json.loads(err_body)
            except Exception:
                err_json = {"error": err_body or str(e)}
            return e.code, err_json
        except Exception as e:
            return 500, {"ok": False, "error": f"Network error connecting to Colab ({self.url}): {str(e)}"}

    def health_check(self) -> bool:
        """Verifies liveness and authentication with the Colab worker."""
        code, resp = self._request("/health")
        return code == 200 and resp.get("ok") is True

    def get_runtime_info(self) -> dict:
        """Queries CPU, GPU (NVIDIA T4/A100), TPU, VRAM, and RAM specs from Colab."""
        code, resp = self._request("/runtime")
        if code == 200 and resp.get("ok"):
            return resp.get("runtime", {})
        raise RuntimeError(f"Failed to query Colab runtime: {resp.get('error', 'Unknown error')}")

    def submit_training_job(
        self,
        model_type: str = "gbdt",
        parameters: Optional[dict] = None,
        script_file: Optional[str] = None,
        script_code: Optional[str] = None
    ) -> str:
        """Submits a training job to Colab. Returns the job_id."""
        params = parameters or {}
        code_str = script_code

        if script_file and os.path.exists(script_file):
            with open(script_file, "r") as f:
                code_str = f.read()

        payload = {
            "model_type": model_type,
            "parameters": params,
            "script_code": code_str
        }

        code, resp = self._request("/train", method="POST", data=payload)
        if code in [200, 202] and resp.get("ok"):
            return resp["job_id"]
        raise RuntimeError(f"Failed to submit training job to Colab: {resp.get('error', 'Unknown error')}")

    def get_job_status(self, job_id: str) -> dict:
        """Checks the status, logs, and artifacts of a specific training job."""
        code, resp = self._request(f"/jobs/{job_id}")
        if code == 200 and resp.get("ok"):
            return resp["job"]
        raise RuntimeError(f"Failed to fetch status for job {job_id}: {resp.get('error', 'Unknown error')}")

    def wait_for_job(self, job_id: str, poll_interval: float = 2.0, verbose: bool = True) -> dict:
        """Polls Colab until job completes while streaming stdout log tails."""
        last_stdout_len = 0
        if verbose:
            print(f"[*] Streaming logs for job {job_id} on Colab...")

        while True:
            jinfo = self.get_job_status(job_id)
            status = jinfo.get("status")
            stdout_tail = jinfo.get("stdout_tail", "")

            if len(stdout_tail) > last_stdout_len:
                new_logs = stdout_tail[last_stdout_len:]
                if verbose:
                    sys.stdout.write(new_logs)
                    sys.stdout.flush()
                last_stdout_len = len(stdout_tail)

            if status in ["completed", "failed"]:
                if verbose:
                    print(f"\n[+] Job {job_id} finished with status: {status.upper()}")
                return jinfo

            time.sleep(poll_interval)

    def download_artifact(self, job_id: str, filename: str, local_path: str) -> bool:
        """Downloads a specific artifact file from Colab to local_path."""
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        target_url = f"{self.url}/jobs/{job_id}/download/{filename}"

        req = urllib.request.Request(target_url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                with open(local_path, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
            return True
        except Exception as e:
            print(f"[-] Error downloading {filename}: {e}")
            return False

    def collect_all_artifacts(self, job_id: str, output_dir: Optional[str] = None) -> List[str]:
        """Downloads all model artifacts and metrics for a completed job into models/."""
        jinfo = self.get_job_status(job_id)
        artifacts = jinfo.get("artifacts", [])
        out_dir = output_dir or os.path.join(ROOT_DIR, "models")
        os.makedirs(out_dir, exist_ok=True)

        downloaded_files = []
        for art in artifacts:
            fname = art["filename"]
            local_target = os.path.join(out_dir, fname)
            print(f"[*] Downloading {fname} ({art.get('size_bytes', 0)} bytes) -> {local_target}")
            if self.download_artifact(job_id, fname, local_target):
                downloaded_files.append(local_target)

        return downloaded_files


if __name__ == "__main__":
    cfg = load_colab_config()
    if cfg:
        print(f"[Colab SDK] Loaded configuration: URL={cfg['url']}")
        client = ColabClient(cfg["url"], cfg["token"])
        if client.health_check():
            print("[+] Connection check SUCCESSful!")
            print(json.dumps(client.get_runtime_info(), indent=2))
        else:
            print("[-] Colab health check failed.")
    else:
        print("[Colab SDK] No saved connection found. Use `colab_cli.py connect` to configure.")
