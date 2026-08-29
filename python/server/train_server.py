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
            self._send(200, {
                "ok": True,
                "last_train": last_train,
                "models_present": sorted(os.listdir(MODELS)) if os.path.isdir(MODELS) else [],
            })
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.startswith("/train/gbdt"):
            result = run_trainer(os.path.join(ROOT, "python", "train", "train_gbdt.py"))
            self._send(200 if result["returncode"] == 0 else 500, result)
        elif self.path.startswith("/train/mlp"):
            self._send(501, {"ok": False, "error": "MLP trainer not yet wired"})
        else:
            self._send(404, {"ok": False, "error": "not found"})


def main():
    port = int(os.environ.get("THE_QUANT_LAB_PORT", "8090"))
    os.makedirs(MODELS, exist_ok=True)
    print(f"[quant-lab] research server on :{port} (offline training only)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()