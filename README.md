# Google Colab LLM + diffusion notebooks

Notebooks for running **LLM and text-to-image** tooling on Colab GPUs (including **Google AI Pro / Colab Pro** L4, A100, and G4 runtimes). Models download once to **Google Drive**, then restore into later sessions. Set Runtime → Change runtime type to the hardware named for the notebook, then Run all.

Notebooks are sorted **runtime → engine → model**: `notebooks/<runtime>/<engine>/<model>.ipynb`, for the
three GPU runtimes **L4**, **A100 80 GB** and **G4**. Open the one for your runtime and model and Run all: it
installs only that engine and serves only that model through one authenticated OpenAI-compatible API.

| Folder | What's in it |
|---|---|
| [`notebooks/l4/`](notebooks/l4/) | L4 (24 GB): one folder per engine, one notebook per model |
| [`notebooks/a100/`](notebooks/a100/) | A100 80 GB (High-RAM) |
| [`notebooks/g4/`](notebooks/g4/) | G4 (RTX PRO 6000 Blackwell, 96 GB), including the G4-only engines (NInfer, Basalt, Strata NVFP4) |
| [`notebooks/cpu/prepare_cpu.ipynb`](notebooks/cpu/prepare_cpu.ipynb) | **Save GPU time:** on a cheap CPU runtime, download a model to Drive and build its engine for the GPU you will use |
| [`notebooks/other/unsloth/`](notebooks/other/unsloth/) | Official Unsloth Studio |

The full list of notebooks is in [`notebooks/README.md`](notebooks/README.md) (and the
[`llm_hub_colab.ipynb`](notebooks/llm_hub_colab.ipynb) index). Engine picks and workstation notes:
[`notebook_code/ENGINE_MATRIX.md`](notebook_code/ENGINE_MATRIX.md). Settings are
`settings/settings_<engine>.json`; weights and builds share Drive.

## Prepare on a CPU runtime, serve on a GPU runtime

Downloading a model and building an engine need no GPU, so do them on a **CPU** runtime with
[`notebooks/cpu/prepare_cpu.ipynb`](notebooks/cpu/prepare_cpu.ipynb): pick the GPU you will use
(L4, A100 or G4), the engine and the model. It downloads the model straight to Drive one file at a time
(resumable; Drive accepts about 750 GB of uploads per day) while it builds or caches the engine for that GPU:
wheels, HyperQwen's venv, llama.cpp / NInfer compiled for the GPU's architecture. The GPU notebook then only
copies them from Drive and loads the model. Without a browser:
`python -m notebook_code prepare --target g4 --engine ninfer --model recommended`.

## Google AI Pro / Colab Pro

Use a Pro runtime with an **L4, A100 80 GB, or G4** GPU. For the A100, pick **High-RAM** in Change runtime type: that gives the 80 GB GPU these notebooks are sized for.

## Code
`notebook_code/` is the Python package behind the hub notebooks (hub server, Drive model store, catalog, backends, tests). Each notebook carries it packed in a cell, so nothing is fetched from GitHub. Regenerate every notebook with `python3 -m notebook_code.make_engine_notebooks`.

## Credit and license
The Studio notebook is from [unslothai/unsloth](https://github.com/unslothai/unsloth); Unsloth Studio is licensed under AGPL-3.0.
llama.cpp is from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
TensorFold is from [ashhart/TensorFold](https://github.com/ashhart/TensorFold); Sparsa from [LyJonathan/Sparsa](https://github.com/LyJonathan/Sparsa); HyperQwen from [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen); NInfer from [Neroued/ninfer](https://github.com/Neroued/ninfer); Basalt from [jesdga95/basalt](https://github.com/jesdga95/basalt); the Strata NVFP4 fork from [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4); ComfyUI from [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI).
Diffusers / Comfy models keep their upstream licenses (Stability AI, Black Forest Labs, etc.); accept gated licenses on Hugging Face and set Colab secret `HF_TOKEN` when needed. Runtimes driven from the command line cannot read Colab Secrets; for those, put the token in a one-line file on your private Drive, `MyDrive/colab_model_server/settings/hf_token.txt` (a Colab secret still wins when present).

## Model and build storage
Notebooks mount Google Drive and keep **all durable state** under `MyDrive/colab_model_server` so later sessions restore instead of re-downloading:

| Path | Contents |
|---|---|
| `models/<id>/` | Weights (GGUF, safetensors, Diffusers / Comfy trees), marked complete only after a full copy |
| `engines/` | llama.cpp tarballs; wheels; TensorFold / Sparsa / ComfyUI / Strata / Maya trees; HyperQwen venv and NInfer build tarballs; `tools/tailscale`, `tools/cloudflared` |
| `settings/settings_<engine>.json` | Per-engine selected model, ctx overrides, Tailscale hostname/port |
| `huggingface/` | Small Hugging Face files only (`HF_HOME`: login token, tokenizer / config lookups); model weights go to `models/` |
| `access/api_key.txt`, `access/tailscale_authkey.txt`, `access/cloudflare_tunnel_token.txt`, `access/endpoint.json` | Private Drive secrets + last public URL (not in git) |
| `logs/` | `install.log`, `hub.log`, `tailscaled.log`, engine logs in `logs/engines/` |
| `outputs/images/`, `outputs/chat/`, `outputs/status/` | Generation results and status snapshots |

A model is downloaded from Hugging Face straight to Drive, once. A model that fits in VRAM + RAM is then read directly from Drive every session (llama.cpp reads it with `--no-mmap` and vLLM with `--safetensors-load-strategy eager`, so the mount is read sequentially). A model that streams weights from SSD during inference (the DeepSeek Engram packs, a GGUF larger than memory) gets its streamed files copied from Drive to the runtime's local disk every session, with the rest read from Drive. Measured on Colab: the first read from Drive runs at about 77 MB/s (23.7 GB in 5 min), and Colab's Drive client caches what it reads on the runtime's local disk (22.6 GB of that 23.7 GB file), so a Drive-read model still occupies local disk and fit checks count it. The first run asks you to authorize Drive access. Set `REFRESH_WHEELS` / `REBUILD_LLAMA` in Drive settings to force updates.

## Remote access
Cell 6 asks how your machine connects (type 1, 2 or 3; Enter keeps the last choice), so nothing in the notebook is edited:

| Choice | Setup | URL | Who can reach it |
|---|---|---|---|
| 1. Tailscale (default) | Tailscale account + auth key | Stable `https://colab-<engine>.<tailnet>.ts.net` | Only your tailnet |
| 2. Cloudflare quick tunnel | None (as in Unsloth Studio's notebook) | New `https://<words>.trycloudflare.com` every session | Anyone with the URL; the API key protects the hub |
| 3. Cloudflare named tunnel | A tunnel in your Cloudflare dashboard + its token | Your stable hostname | Public; the API key protects the hub |

Keys: add `TAILSCALE_AUTHKEY` (and `HF_TOKEN`, `CLOUDFLARE_TUNNEL_TOKEN`) once in Colab's Secrets panel (key icon, left sidebar) and no notebook asks for them. Otherwise cell 6 asks in a hidden input box and keeps the key on **your private Google Drive** (`MyDrive/colab_model_server/access/tailscale_authkey.txt`, which the Drive app on a phone opens); type `1 new` at the tunnel prompt to replace it. The hub API key is also Drive-only (`access/api_key.txt`). Nothing secret is ever written into a notebook or this repository.

The Cloudflare link is printed only after the tunnel has registered and the hub answers through it. Your harness (dsh or any OpenAI client) normally runs on your own machine. Optionally, cell 6b runs dsh inside the notebook instead, already pointed at the runtime's models, and opens its web UI through Colab's own proxy (only you can open it; dsh also requires the token in its link). Its conversations are kept on Drive in `dsh/`, and the agent works in `workspace/`. The T4 / TPU notebooks offer `tailscale` and `cloudflare` too.

Fit checks use the GPU's VRAM, system RAM (for models that spill into it) and the local disk (streamed files, plus the Drive client's cache of everything read from Drive); the model table says why a model is excluded. Streamed files go to the larger of `/content` and `/mnt/local-scratch` (the A100 80 GB's extra ~350 GB NVMe disk).

Each notebook downloads and serves **one model**, picked in its `MODEL` dropdown (`recommended` = the first recommended model that fits the GPU). With `SERVE_ONLY_SELECTED` (on by default) the hub serves only that model, so a harness request for another model cannot start a large download.

Builds are cached too: the first run builds/downloads and saves them; later runs restore them. Set `REBUILD_LLAMA` or `REFRESH_WHEELS` in the notebook to update. A cache is tied to the Python version (and, for llama.cpp, the GPU arch), so if Colab changes its image the notebook rebuilds automatically.
