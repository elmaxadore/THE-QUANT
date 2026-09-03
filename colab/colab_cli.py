#!/usr/bin/env python3
"""THE QUANT — Google Colab CLI & Interactive Remote Trainer.

Clear, user-friendly command-line tool to:
  1. `connect`    - Pair THE-QUANT with your Google Colab runtime URL & Token.
  2. `info`       - View Colab GPU/TPU/CPU hardware specs and connection status.
  3. `runtimes`   - Display available model acceleration choices & recommended settings.
  4. `train`      - Dispatch training scripts (GBDT, PyTorch MLP, Custom script) to Colab.
  5. `collect`    - Download trained `.onnx` models and metrics directly into `THE-QUANT/models/`.
  6. `interactive`- Guided step-by-step wizard for end-to-end training and artifact collection.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Ensure project root is in import path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from colab.colab_bridge import ColabClient, save_colab_config, load_colab_config


def get_client() -> ColabClient:
    cfg = load_colab_config()
    if not cfg:
        print("[!] No Colab connection configured yet.")
        print("[!] Run: python colab/colab_cli.py connect --url <COLAB_URL> --token <TOKEN>")
        sys.exit(1)

    client = ColabClient(cfg["url"], cfg["token"])
    if not client.health_check():
        print(f"[!] Unable to reach Colab runtime at {cfg['url']}")
        print("[!] Please check if the Colab worker notebook is currently running.")
        sys.exit(1)
    return client


def cmd_connect(args):
    url = args.url.strip()
    token = args.token.strip()

    print(f"[*] Testing connection to Colab at {url}...")
    client = ColabClient(url, token)
    if client.health_check():
        save_colab_config(url, token)
        print("[+] SUCCESS! Successfully connected and authenticated with Colab worker.")
        hw = client.get_runtime_info()
        print("\n--- Colab Hardware Summary ---")
        print(f"Accelerator      : {hw.get('accelerator_type')}")
        print(f"Recommended Mode : {hw.get('recommended_runtime')}")
        if hw.get("gpu"):
            gpu = hw["gpu"]
            print(f"GPU Model        : {gpu.get('name')} ({gpu.get('vram_gb')} GB VRAM)")
        elif hw.get("tpu"):
            print(f"TPU Status       : Active")
        print(f"System RAM       : {hw.get('ram_gb')} GB")
        print(f"CPU Cores        : {hw.get('cpu_count')}")
    else:
        print("[-] Connection failed. Please verify the URL and secret token.")
        sys.exit(1)


def cmd_info(args):
    client = get_client()
    hw = client.get_runtime_info()
    print("=" * 65)
    print("           THE QUANT — COLAB RUNTIME ENVIRONMENT INFO")
    print("=" * 65)
    print(f"Status           : Connected & Verified")
    print(f"Colab URL        : {client.url}")
    print(f"Accelerator      : {hw.get('accelerator_type')}")
    print(f"Recommended Mode : {hw.get('recommended_runtime')}")
    print(f"System RAM       : {hw.get('ram_gb')} GB")
    print(f"CPU Cores        : {hw.get('cpu_count')}")
    print(f"Python Version   : {hw.get('python_version')}")

    if hw.get("gpu"):
        gpu = hw["gpu"]
        print("-" * 65)
        print("NVIDIA GPU Details:")
        print(f"  Device Name    : {gpu.get('name')}")
        print(f"  VRAM           : {gpu.get('vram_gb')} GB")
        print(f"  CUDA Version   : {gpu.get('cuda_version')}")
        print(f"  PyTorch        : {gpu.get('torch_version')}")
    elif hw.get("tpu"):
        print("-" * 65)
        print("Google TPU Details:")
        print(f"  TPU Address    : {hw['tpu'].get('address')}")
    print("=" * 65)


def cmd_runtimes(args):
    client = get_client()
    hw = client.get_runtime_info()
    acc = hw.get("accelerator_type", "CPU")

    print("\n" + "=" * 70)
    print("        AVAILABLE TRAINING RUNTIME OPTIONS & HARDWARE MATCHING")
    print("=" * 70)
    print(f"Current Colab Environment Hardware: {acc}\n")

    runtimes = [
        {
            "id": "gbdt",
            "name": "Gradient Boosted Decision Trees (GBDT)",
            "description": "Fast tree ensemble model. Produces lightweight ONNX for ultra low-latency Rust inference.",
            "recommended_acc": ["CPU", "GPU_STANDARD", "GPU_HIGH_PERFORMANCE"],
            "default_params": "n_samples=50,000 | n_estimators=120 | max_depth=4"
        },
        {
            "id": "mlp",
            "name": "PyTorch Neural Network (MLP)",
            "description": "Deep SiLU multi-layer perceptron with dropout and batch normalization. Fully GPU-accelerated.",
            "recommended_acc": ["GPU_STANDARD", "GPU_HIGH_PERFORMANCE", "TPU"],
            "default_params": "n_samples=100,000 | epochs=30 | batch_size=512 | lr=0.001"
        },
        {
            "id": "custom",
            "name": "Custom Python Script Upload",
            "description": "Execute any user-provided PyTorch/Scikit-Learn/TensorFlow script on Colab GPU/TPU.",
            "recommended_acc": ["CPU", "GPU_STANDARD", "GPU_HIGH_PERFORMANCE", "TPU"],
            "default_params": "Custom user script payload"
        }
    ]

    for r in runtimes:
        is_rec = acc in r["recommended_acc"]
        rec_badge = " [RECOMMENDED FOR CURRENT ACCELERATOR]" if is_rec else ""
        print(f"[{r['id'].upper()}] - {r['name']}{rec_badge}")
        print(f"  Description  : {r['description']}")
        print(f"  Defaults     : {r['default_params']}")
        print("-" * 70)


def cmd_train(args):
    client = get_client()

    params = {
        "n_samples": args.n_samples,
        "n_estimators": args.n_estimators,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
    }
    if args.params_json:
        # Merge free-form pipeline parameters (e.g. {"symbols": ..., "start": ...})
        # — forwarded as the TRAIN_PARAMS env var inside the Colab job.
        from_colab_json = json.loads(args.params_json)
        params.update(from_colab_json)

    script_path = args.script if args.type == "custom" else None
    if args.type == "custom" and not script_path:
        print("[!] For custom model type, you must provide --script <path_to_python_file>")
        sys.exit(1)

    print(f"[*] Submitting '{args.type.upper()}' training job to Colab runtime...")
    job_id = client.submit_training_job(
        model_type=args.type,
        parameters=params,
        script_file=script_path
    )

    print(f"[+] Job {job_id} submitted! Monitoring live execution...")
    jinfo = client.wait_for_job(job_id, verbose=True)

    if jinfo.get("status") == "completed":
        print(f"\n[+] Training job {job_id} completed successfully!")
        if args.auto_collect:
            print("[*] Automatically collecting output model files into models/...")
            collected = client.collect_all_artifacts(job_id)
            print(f"[+] Downloaded {len(collected)} artifact(s) to THE-QUANT/models/")
        else:
            print(f"[*] Run `python colab/colab_cli.py collect --job-id {job_id}` to download outputs.")
    else:
        print(f"\n[-] Job {job_id} failed with exit code {jinfo.get('returncode')}.")
        sys.exit(1)


def cmd_collect(args):
    client = get_client()

    job_id = args.job_id
    if not job_id:
        # Pick last job
        jobs = client._request("/jobs")[1].get("jobs", {})
        if not jobs:
            print("[!] No training jobs found on Colab.")
            sys.exit(1)
        job_id = list(jobs.keys())[-1]
        print(f"[*] Defaulting to latest job: {job_id}")

    print(f"[*] Collecting model artifacts for job {job_id}...")
    out_dir = args.output_dir or os.path.join(ROOT_DIR, "models")
    collected = client.collect_all_artifacts(job_id, output_dir=out_dir)

    # Route reports (.json) to reports/, models (.onnx) to models/ so the repo
    # keeps "collect output only" clean.
    reports_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    split_collected = []
    for path in collected:
        if path.endswith(".json") and os.path.dirname(path) != reports_dir:
            dest = os.path.join(reports_dir, os.path.basename(path))
            os.replace(path, dest)
            split_collected.append(dest)
        else:
            split_collected.append(path)

    print(f"\n[+] Successfully saved {len(split_collected)} output file(s):")
    for path in split_collected:
        print(f"  -> {path}")


def cmd_interactive(args):
    print("=" * 70)
    print("          THE QUANT — INTERACTIVE COLAB TRAINING WIZARD")
    print("=" * 70)

    cfg = load_colab_config()
    if cfg:
        print(f"[*] Saved Colab URL: {cfg['url']}")
        use_saved = input("Use saved Colab connection? [Y/n]: ").strip().lower()
        if use_saved not in ["", "y", "yes"]:
            cfg = None

    if not cfg:
        url = input("Enter Colab Public Tunnel URL (e.g., https://xxxx.trycloudflare.com): ").strip()
        token = input("Enter Secret Auth Token: ").strip()
        save_colab_config(url, token)
        cfg = {"url": url, "token": token}

    client = ColabClient(cfg["url"], cfg["token"])
    print("\n[*] Connecting to Colab...")
    if not client.health_check():
        print("[-] Connection failed. Please check Colab worker server.")
        return

    hw = client.get_runtime_info()
    print(f"\n[+] Connected! Hardware Accelerator: {hw.get('accelerator_type')} ({hw.get('recommended_runtime')})")

    print("\nSelect Training Runtime & Model Type:")
    print("  1) GBDT (Gradient Boosted Trees - Super fast, lightweight ONNX)")
    print("  2) PyTorch MLP (Deep Neural Net - GPU/TPU accelerated)")
    print("  3) Custom Python Script")
    choice = input("Choice [1-3, default=1]: ").strip()

    if choice == "2":
        model_type = "mlp"
        epochs = int(input("Epochs [default=30]: ") or "30")
        samples = int(input("Training Samples [default=100000]: ") or "100000")
        params = {"epochs": epochs, "n_samples": samples, "batch_size": 512, "learning_rate": 0.001}
        script_file = None
    elif choice == "3":
        model_type = "custom"
        script_file = input("Enter local path to custom Python script: ").strip()
        params = {}
    else:
        model_type = "gbdt"
        samples = int(input("Training Samples [default=50000]: ") or "50000")
        estimators = int(input("Trees / Estimators [default=120]: ") or "120")
        params = {"n_samples": samples, "n_estimators": estimators, "max_depth": 4}
        script_file = None

    print(f"\n[*] Submitting {model_type.upper()} training job to Colab...")
    job_id = client.submit_training_job(model_type=model_type, parameters=params, script_file=script_file)

    jinfo = client.wait_for_job(job_id, verbose=True)

    if jinfo.get("status") == "completed":
        print(f"\n[+] Training job {job_id} COMPLETED!")
        collect_now = input("Collect and download `.onnx` models into THE-QUANT/models/? [Y/n]: ").strip().lower()
        if collect_now in ["", "y", "yes"]:
            collected = client.collect_all_artifacts(job_id)
            print(f"[+] Downloaded {len(collected)} model files directly into THE-QUANT/models/")
    else:
        print(f"[-] Training job failed: {jinfo.get('returncode')}")


def cmd_enqueue(args):
    """Queue a job on the colab-jobs branch — any running agent picks it up."""
    from colab import persist
    import datetime
    script = args.script or ""
    params = {}
    if args.params_json:
        try:
            params = json.loads(args.params_json)
        except json.JSONDecodeError as e:
            print(f"[-] --params-json is not valid JSON: {e}")
            sys.exit(1)
    slug = os.path.splitext(os.path.basename(script or "train"))[0]
    job_name = args.name or \
        f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    spec = {
        "script": script,
        "params": params,
        "status": "pending",
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "created_by": os.environ.get("USER", "local"),
    }
    persist.enqueue_job(ROOT_DIR, job_name, spec)
    print(f"[+] Job '{job_name}' queued. Any running agent "
          f"(Colab/Kaggle/Codespaces) will pick it up within "
          f"~{30}s of its poll loop.")
    print(f"[+] Recover results any time (even days later) with:")
    print(f"      python3 colab/colab_cli.py pull")


def cmd_status(args):
    """Show queued/claimed/completed jobs on the colab-jobs branch."""
    subprocess.run(["git", "-C", ROOT_DIR, "fetch", "origin",
                    persist.JOBS_BRANCH, persist.ARTIFACTS_BRANCH],
                   capture_output=True)
    for branch, label in [(persist.JOBS_BRANCH, "JOB QUEUE"),
                          (persist.ARTIFACTS_BRANCH, "ARTIFACTS")]:
        print(f"--- {label} (origin/{branch}) ---")
        subprocess.run(["git", "-C", ROOT_DIR, "log", "--oneline", "-8",
                        f"origin/{branch}"], check=False)


def cmd_pull(args):
    """Recover every artifact from the colab-artifacts branch into the repo."""
    from colab import persist
    subprocess.run(["git", "-C", ROOT_DIR, "fetch", "origin",
                    persist.ARTIFACTS_BRANCH], capture_output=True)
    files = persist.list_artifacts(ROOT_DIR)
    if not files:
        print("[!] No artifacts found on origin/"
              f"{persist.ARTIFACTS_BRANCH} yet.")
        return
    routes = [(".json", os.path.join(ROOT_DIR, "reports")),
              (".onnx", os.path.join(ROOT_DIR, "models")),
              (".csv", os.path.join(ROOT_DIR, "python", "data",
                                    "histdata"))]
    counts = {}
    for rel in files:
        dest_dir = None
        for ext, d in routes:
            if rel.endswith(ext):
                dest_dir = d
                break
        if dest_dir is None:
            continue
        os.makedirs(dest_dir, exist_ok=True)
        content = subprocess.run(
            ["git", "-C", ROOT_DIR, "show",
             f"origin/{persist.ARTIFACTS_BRANCH}:{rel}"],
            capture_output=True).stdout
        with open(os.path.join(dest_dir, os.path.basename(rel)), "wb") as f:
            f.write(content)
        key = dest_dir
        counts[key] = counts.get(key, 0) + 1
    print("[+] Restored artifacts:")
    for d, n in counts.items():
        print(f"    {n:3d} file(s) -> {os.path.relpath(d, ROOT_DIR)}/")
    print("[+] Data loss-proof: originals remain on origin/"
          f"{persist.ARTIFACTS_BRANCH} and are never deleted.")


def main():
    parser = argparse.ArgumentParser(description="THE QUANT Google Colab Launcher & Bridge")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # connect
    p_conn = subparsers.add_parser("connect", help="Connect and save Colab connection profile")
    p_conn.add_argument("--url", required=True, help="Public Colab Tunnel URL")
    p_conn.add_argument("--token", required=True, help="Secret Auth Token")

    # info
    p_info = subparsers.add_parser("info", help="View Colab hardware accelerator (GPU/TPU/CPU) info")

    # runtimes
    p_rt = subparsers.add_parser("runtimes", help="List model runtime options for Colab hardware")

    # train
    p_train = subparsers.add_parser("train", help="Run a training script on Colab")
    p_train.add_argument("--type", choices=["gbdt", "mlp", "custom"], default="gbdt", help="Model type")
    p_train.add_argument("--script", help="Path to custom Python training script")
    p_train.add_argument("--n-samples", type=int, default=50000, help="Number of samples")
    p_train.add_argument("--n-estimators", type=int, default=120, help="Trees for GBDT")
    p_train.add_argument("--epochs", type=int, default=30, help="Epochs for MLP")
    p_train.add_argument("--batch-size", type=int, default=512, help="Batch size for MLP")
    p_train.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    p_train.add_argument("--max-depth", type=int, default=4, help="Tree max depth")
    p_train.add_argument("--no-collect", dest="auto_collect", action="store_false", help="Skip auto download of artifacts")
    p_train.add_argument("--params-json", help="Extra pipeline params as JSON, e.g. "
                        "'{\"symbols\": \"eurusd,xauusd\", \"start\": \"2023-01-01\"}'")

    # collect
    p_coll = subparsers.add_parser("collect", help="Download trained ONNX model artifacts from Colab")
    p_coll.add_argument("--job-id", help="Job ID (defaults to latest job)")
    p_coll.add_argument("--output-dir", help="Directory to save downloaded ONNX models (default: models/)")

    # interactive
    p_inter = subparsers.add_parser("interactive", help="Guided interactive wizard")

    # enqueue (agent mode — no tunnel needed)
    p_enq = subparsers.add_parser("enqueue", help="Queue a job for any running remote agent (Colab/Kaggle/Codespaces)")
    p_enq.add_argument("--script", help="Repo-relative job script, e.g. colab/jobs/download_data.py")
    p_enq.add_argument("--params-json", help='Params as JSON, e.g. \'{"symbols": "eurusd,xauusd", "start": "2025-01-01"}\'')
    p_enq.add_argument("--name", help="Optional job name")

    # status
    p_stat = subparsers.add_parser("status", help="Show remote job queue and artifact history")

    # pull
    p_pull = subparsers.add_parser("pull", help="Recover all artifacts from the GitHub artifacts branch (data-loss-proof)")

    args = parser.parse_args()

    if args.command == "connect":
        cmd_connect(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "runtimes":
        cmd_runtimes(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "enqueue":
        cmd_enqueue(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "pull":
        cmd_pull(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    else:
        # Default to interactive if no args given
        cmd_interactive(args)


if __name__ == "__main__":
    main()
