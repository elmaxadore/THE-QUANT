# ☁️ THE-QUANT — Google Colab Cloud Training Integration

This module allows **THE-QUANT** to seamlessly connect to a **Google Colab** runtime (CPU, NVIDIA T4 / A100 GPU, or TPU), choose model runtimes, execute PyTorch / GBDT training scripts on Colab's cloud hardware, stream live logs, and automatically collect exported `.onnx` models into `models/`.

---

## 📁 Architecture Overview

```text
Google Colab Runtime (Cloud GPU/TPU)            Local THE-QUANT Project
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│ [quant_colab_runner.ipynb]           │        │ [colab_cli.py] (CLI / Interactive)  │
│              │                       │        │                 │                    │
│ [colab_worker.py] (Fast REST Server) │◄───────┼─ Cloudflare     │                    │
│  - Detects GPU (T4/A100), TPU, RAM   │ Tunnel │   Tunnel URL    │                    │
│  - Executes training scripts         │  (SSL) │                 ▼                    │
│  - Exports .onnx & metrics           │        │ [colab_bridge.py] (Python SDK)       │
└──────────────────────────────────────┘        └──────────────────┬───────────────────┘
                                                                   │ (Downloads .onnx)
                                                                   ▼
                                                        [models/latest.onnx]
                                                                   │ (Inference)
                                                                   ▼
                                                        [Rust Core (src/onnx.rs)]
```

---

## 🚀 Quick Start Guide

### Step 1: Start the Colab Worker
1. Open [`colab/quant_colab_runner.ipynb`](file:///home/maxmillian/.cline/data/workspaces/chat/THE-QUANT/colab/quant_colab_runner.ipynb) in Google Colab.
2. Select your desired runtime accelerator in Colab:
   - Menu: **Runtime** -> **Change runtime type** -> Select **GPU** (T4 / A100) or **TPU**.
3. Run all cells in the notebook.
4. The notebook will display your **Public Tunnel URL** and **Secret Auth Token**:
   ```text
   📍 Public Tunnel URL : https://xxxxxxxx.trycloudflare.com
   🔑 Secret Token     : quant-colab-secret-key
   ```

### Step 2: Connect Local THE-QUANT
In your local `THE-QUANT` directory, run:
```bash
python colab/colab_cli.py connect --url https://xxxxxxxx.trycloudflare.com --token quant-colab-secret-key
```

### Step 3: Inspect Remote Hardware Runtime
Verify GPU/TPU acceleration and view recommended model choices:
```bash
python colab/colab_cli.py info
python colab/colab_cli.py runtimes
```

### Step 4: Run Remote Training & Collect Output Models
Train a PyTorch deep neural network (or GBDT ensemble) on Colab GPU and collect `.onnx` outputs automatically:

```bash
# Train PyTorch MLP model on Colab GPU and download latest.onnx
python colab/colab_cli.py train --type mlp --epochs 30 --n-samples 100000

# Or train GBDT tree ensemble
python colab/colab_cli.py train --type gbdt --n-estimators 150

# Or execute a custom local python training script on Colab GPU
python colab/colab_cli.py train --type custom --script path/to/my_script.py
```

### 🧙 Step 5: Or Use the Interactive Guided Wizard
```bash
python colab/colab_cli.py interactive
```

---

## 🛠️ CLI Reference (`colab/colab_cli.py`)

| Command | Subcommand | Description |
|---|---|---|
| `connect` | `--url <URL> --token <TOKEN>` | Pairs local THE-QUANT with Colab URL & saves profile |
| `info` | | Displays Colab GPU name, VRAM, TPU status, RAM, CPU cores |
| `runtimes` | | Shows available model acceleration choices (GBDT, MLP, Custom) |
| `train` | `--type <gbdt\|mlp\|custom>` | Dispatches training job to Colab, streams logs, auto-downloads ONNX |
| `collect` | `--job-id <id>` | Downloads `.onnx` models & `metrics.json` into `models/` |
| `interactive` | | Interactive terminal wizard |

---

## 🧪 Verifying the Installation
Run the unit test suite to verify client/server protocol, hardware detection, and artifact collection:
```bash
python3 -m unittest colab/tests/test_colab_bridge.py
```
