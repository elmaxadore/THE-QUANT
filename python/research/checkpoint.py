#!/usr/bin/env python3
"""THE QUANT — stage-based checkpointing for resumable training pipelines.

On free Colab, runtimes die without warning (~90 min session limit). This module
makes the training pipeline SURVIVE disconnections by checkpointing after every
stage:

    download -> features -> train_mlp -> train_xgb -> backtest -> grid_search -> export

Each stage's output is saved to a local checkpoint directory. On restart, the
pipeline reads the checkpoint, sees which stages are done, and skips to the
next incomplete one. Combined with periodic pushes to the colab-artifacts
branch, this means a job can survive ANY number of Colab disconnections and
always make forward progress.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

STAGES = [
    "download", "features", "train_mlp", "train_xgb",
    "backtest", "grid_search", "export",
]


class Checkpoint:
    """Manages stage-based progress for a single job."""

    def __init__(self, job_name: str, checkpoint_dir: str,
                 auto_push: bool = False, repo_dir: str = "",
                 token: str = ""):
        self.job_name = job_name
        self.checkpoint_dir = os.path.join(checkpoint_dir, job_name)
        self.auto_push = auto_push
        self.repo_dir = repo_dir
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.manifest_path = os.path.join(self.checkpoint_dir, "manifest.json")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if os.path.isfile(self.manifest_path):
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"job_name": self.job_name, "stages": {},
                "created_at": time.time(), "updated_at": time.time()}

    def _save_manifest(self):
        self.manifest["updated_at"] = time.time()
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(tmp, self.manifest_path)

    def is_done(self, stage: str) -> bool:
        return stage in self.manifest.get("stages", {}) and \
               self.manifest["stages"][stage].get("status") == "done"

    def mark_done(self, stage: str, metadata: Optional[Dict] = None):
        if "stages" not in self.manifest:
            self.manifest["stages"] = {}
        self.manifest["stages"][stage] = {
            "status": "done", "completed_at": time.time(),
            "metadata": metadata or {}}
        self._save_manifest()

    def mark_in_progress(self, stage: str):
        if "stages" not in self.manifest:
            self.manifest["stages"] = {}
        self.manifest["stages"][stage] = {
            "status": "in_progress", "started_at": time.time()}
        self._save_manifest()

    def save(self, stage: str, data: Any, fmt: str = "auto") -> str:
        stage_dir = os.path.join(self.checkpoint_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        if fmt == "auto":
            if isinstance(data, np.ndarray):
                fmt = "numpy"
            elif isinstance(data, (dict, list)):
                fmt = "json"
            elif isinstance(data, pd.DataFrame):
                fmt = "pickle"  # pickle avoids pyarrow dependency
            else:
                fmt = "pickle"
        if fmt == "numpy":
            path = os.path.join(stage_dir, "data.npz")
            if isinstance(data, dict):
                np.savez_compressed(path, **data)
            else:
                np.savez_compressed(path, data=data)
        elif fmt == "parquet":
            path = os.path.join(stage_dir, "data.parquet")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, pd.DataFrame):
                        v.to_parquet(os.path.join(stage_dir, f"{k}.parquet"),
                                     index=False)
                path = stage_dir
            else:
                data.to_parquet(path, index=False)
        elif fmt == "json":
            path = os.path.join(stage_dir, "data.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif fmt == "pickle":
            import pickle
            path = os.path.join(stage_dir, "data.pkl")
            with open(path, "wb") as f:
                pickle.dump(data, f)
        return path

    def load(self, stage: str) -> Any:
        stage_dir = os.path.join(self.checkpoint_dir, stage)
        if not os.path.isdir(stage_dir):
            raise FileNotFoundError(f"No checkpoint for stage '{stage}'")
        parquet_files = [f for f in os.listdir(stage_dir)
                         if f.endswith(".parquet")]
        if parquet_files:
            if len(parquet_files) == 1:
                return pd.read_parquet(os.path.join(stage_dir, parquet_files[0]))
            return {f.replace(".parquet", ""): pd.read_parquet(
                        os.path.join(stage_dir, f))
                    for f in parquet_files}
        npz_path = os.path.join(stage_dir, "data.npz")
        if os.path.isfile(npz_path):
            d = np.load(npz_path)
            if len(d.files) == 1 and "data" in d.files:
                return d["data"]
            return {k: d[k] for k in d.files}
        json_path = os.path.join(stage_dir, "data.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                return json.load(f)
        import pickle
        pkl_path = os.path.join(stage_dir, "data.pkl")
        if os.path.isfile(pkl_path):
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"Could not find checkpoint data in {stage_dir}")

    def get_progress(self) -> Dict[str, Any]:
        done = sum(1 for s in STAGES if self.is_done(s))
        return {
            "total_stages": len(STAGES),
            "completed_stages": done,
            "stages": self.manifest.get("stages", {}),
            "next_stage": next((s for s in STAGES if not self.is_done(s)), None),
            "is_complete": done == len(STAGES),
        }

    def push_to_artifacts(self, subdir: str = "") -> Optional[str]:
        if not self.auto_push or not self.repo_dir or not self.token:
            return None
        try:
            import persist
            files = []
            for root, _, filenames in os.walk(self.checkpoint_dir):
                for fn in filenames:
                    files.append(os.path.join(root, fn))
            if files:
                sha = persist.push_artifacts(
                    self.repo_dir, files,
                    message=f"checkpoint: {self.job_name}",
                    subdir=f"checkpoints/{self.job_name}/{subdir}" if subdir
                           else f"checkpoints/{self.job_name}",
                    token=self.token)
                return sha
        except Exception as e:
            print(f"[checkpoint] push failed: {e}", flush=True)
        return None

    @staticmethod
    def restore_from_artifacts(repo_dir: str, job_name: str,
                                checkpoint_dir: str,
                                token: str = "") -> bool:
        token = token or os.environ.get("GITHUB_TOKEN", "")
        if not token:
            return False
        try:
            import persist
            push_url = persist._push_url(repo_dir, token)
            with tempfile.TemporaryDirectory(prefix="dq_ckpt_restore_") as tmp:
                r = persist._run(["git", "clone", "--depth", "1",
                                  "--branch", persist.ARTIFACTS_BRANCH,
                                  push_url, tmp], check=False)
                if r.returncode != 0:
                    return False
                src = os.path.join(tmp, "checkpoints", job_name)
                if os.path.isdir(src):
                    dst = os.path.join(checkpoint_dir, job_name)
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    return True
        except Exception as e:
            print(f"[checkpoint] restore failed: {e}", flush=True)
        return False
        return path
        return False


class TrainingCheckpoint:
    """Checkpoint for training loops (save model weights every N epochs)."""

    def __init__(self, job_name: str, stage_name: str,
                 checkpoint_dir: str, every_n_epochs: int = 10,
                 auto_push: bool = False, repo_dir: str = "",
                 token: str = ""):
        self.job_name = job_name
        self.stage_name = stage_name
        self.every_n_epochs = every_n_epochs
        self.auto_push = auto_push
        self.repo_dir = repo_dir
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.dir = os.path.join(checkpoint_dir, job_name, stage_name)
        os.makedirs(self.dir, exist_ok=True)
        self.manifest_path = os.path.join(self.dir, "training_manifest.json")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        if os.path.isfile(self.manifest_path):
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"epochs_done": 0, "best_loss": float("inf"),
                "history": [], "checkpoints": []}

    def _save_manifest(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(tmp, self.manifest_path)

    def should_save(self, epoch: int) -> bool:
        return epoch > 0 and epoch % self.every_n_epochs == 0

    def get_resume_epoch(self) -> int:
        return self.manifest.get("epochs_done", 0)

    def get_best_loss(self) -> float:
        return self.manifest.get("best_loss", float("inf"))

    def save(self, epoch: int, model_state: dict, optimizer_state: dict,
             val_loss: float, extra: Optional[Dict] = None):
        import pickle
        arrays = {}
        for k, v in model_state.items():
            if hasattr(v, 'cpu'):
                arrays[k] = v.cpu().numpy()
            elif isinstance(v, np.ndarray):
                arrays[k] = v
        ckpt_path = os.path.join(self.dir, f"epoch_{epoch}.npz")
        np.savez_compressed(ckpt_path, **arrays)
        opt_path = os.path.join(self.dir, f"optimizer_{epoch}.pkl")
        with open(opt_path, "wb") as f:
            pickle.dump(optimizer_state, f)
        self.manifest["epochs_done"] = epoch
        self.manifest["best_loss"] = min(self.manifest["best_loss"], val_loss)
        self.manifest["history"].append({
            "epoch": epoch, "val_loss": val_loss, "timestamp": time.time()})
        self.manifest["checkpoints"].append(epoch)
        self._save_manifest()
        latest_path = os.path.join(self.dir, "latest.txt")
        with open(latest_path, "w") as f:
            f.write(str(epoch))
        if self.auto_push:
            self._push_checkpoint(epoch)

    def load_latest(self) -> Optional[Dict]:
        latest_path = os.path.join(self.dir, "latest.txt")
        if not os.path.isfile(latest_path):
            return None
        with open(latest_path) as f:
            epoch = int(f.read().strip())
        ckpt_path = os.path.join(self.dir, f"epoch_{epoch}.npz")
        if not os.path.isfile(ckpt_path):
            return None
        d = np.load(ckpt_path)
        model_state = {k: d[k] for k in d.files}
        import pickle
        opt_path = os.path.join(self.dir, f"optimizer_{epoch}.pkl")
        optimizer_state = {}
        if os.path.isfile(opt_path):
            with open(opt_path, "rb") as f:
                optimizer_state = pickle.load(f)
        return {"epoch": epoch, "model_state": model_state,
                "optimizer_state": optimizer_state}

    def _push_checkpoint(self, epoch: int):
        if not self.repo_dir or not self.token:
            return
        try:
            import persist
            files = [
                os.path.join(self.dir, f"epoch_{epoch}.npz"),
                os.path.join(self.dir, f"optimizer_{epoch}.pkl"),
                self.manifest_path,
                os.path.join(self.dir, "latest.txt"),
            ]
            persist.push_artifacts(
                self.repo_dir, files,
                message=f"training_ckpt: {self.job_name} epoch {epoch}",
                subdir=f"checkpoints/{self.job_name}/{self.stage_name}",
                token=self.token)
        except Exception as e:
            print(f"[training_ckpt] push failed: {e}", flush=True)


def should_resume_job(job_name: str, checkpoint_dir: str,
                      repo_dir: str = "", token: str = "") -> tuple:
    local_ckpt = os.path.join(checkpoint_dir, job_name)
    if os.path.isfile(os.path.join(local_ckpt, "manifest.json")):
        return True, local_ckpt
    if repo_dir and token:
        if Checkpoint.restore_from_artifacts(repo_dir, job_name,
                                              checkpoint_dir, token):
            return True, local_ckpt
    return False, None