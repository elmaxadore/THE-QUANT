#!/usr/bin/env python3
"""THE QUANT — offline research/training server (HYBRID).

This server is NOT in the live trading hot path. It offers the training and
research endpoints used during evolution cycles:

    GET  /health            -> liveness probe
    POST /train/gbdt        -> run the GBDT trainer, produce models/latest.onnx
    POST /train/mlp         -> run the (small) MLP trainer, produce models/mlp.onnx
    GET  /status            -> report last training run, model info

The Rust core calls this over HTTP *only* during offline evolution (Tier-3+).
On Tier-1/2 the Rust side skips it entirely and uses the built-in rule model.
"""
import datetime
import json
import os
import subprocess
import sys

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Repo root: this file lives at <repo>/python/server/train_server.py
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "models")

last_train = None


def run_trainer(script: str) -> dict:
    global last_train
    started = datetime.datetime.utcnow().isoformat()
    r = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)
    last_train = {
        "script": script,
        "started": started,
        "finished": datetime.datetime.utcnow().isoformat(),
        "returncode": r.returncode,
        "stdout_tail": r.stdout[-2000:],
        "stderr_tail": r.stderr[-2000:],
    }
    return last_train


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "role": "offline-training-only"})
        elif self.path.startswith("/status"):
            colab_info = None
            try:
                from colab.colab_bridge import load_colab_config, ColabClient
                cfg = load_colab_config()
                if cfg:
                    c = ColabClient(cfg["url"], cfg["token"])
                    if c.health_check():
                        colab_info = c.get_runtime_info()
            except Exception:
                pass

            self._send(200, {
                "ok": True,
                "last_train": last_train,
                "models_present": sorted(os.listdir(MODELS)) if os.path.isdir(MODELS) else [],
                "colab_runtime": colab_info,
            })
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.startswith("/train/gbdt"):
            result = run_trainer(os.path.join(ROOT, "python", "train", "train_gbdt.py"))
            self._send(200 if result["returncode"] == 0 else 500, result)
        elif self.path.startswith("/train/mlp"):
            result = run_trainer(os.path.join(ROOT, "python", "train", "train_mlp.py"))
            self._send(200 if result["returncode"] == 0 else 500, result)
        elif self.path.startswith("/train/colab"):
            try:
                from colab.colab_bridge import load_colab_config, ColabClient
                cfg = load_colab_config()
                if not cfg:
                    self._send(400, {"ok": False, "error": "Colab not configured. Run `python colab/colab_cli.py connect` first."})
                    return
                client = ColabClient(cfg["url"], cfg["token"])
                job_id = client.submit_training_job(model_type="mlp")
                jinfo = client.wait_for_job(job_id, verbose=False)
                if jinfo.get("status") == "completed":
                    client.collect_all_artifacts(job_id)
                    self._send(200, {"ok": True, "job_id": job_id, "status": "completed"})
                else:
                    self._send(500, {"ok": False, "job_id": job_id, "error": "Colab job failed", "detail": jinfo})
            except Exception as e:
                self._send(500, {"ok": False, "error": str(e)})
        else:
            self._send(404, {"ok": False, "error": "not found"})


def main():
    port = int(os.environ.get("THE_QUANT_LAB_PORT", "8090"))
    os.makedirs(MODELS, exist_ok=True)
    print(f"[quant-lab] research server on :{port} (offline training only)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()