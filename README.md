# Google Colab LLM + diffusion notebooks

Notebooks for running **LLM and text-to-image** tooling on Colab GPUs (including **Google AI Pro / Colab Pro** L4 (RTX 4070 Ti class), A100, and G4 (RTX PRO 6000) runtimes). Models download once to **Google Drive**, then restore into later sessions. Set Runtime → Change runtime type to the hardware named for the notebook, then Run all.

Notebooks are sorted **runtime → engine → model**: `notebooks/<runtime>/<engine>/<model>.ipynb` (`l4-rtx4070ti/`, `a100/`, `g4-rtx-pro6000/`), for the
three GPU runtimes **L4** (RTX 4070 Ti class), **A100 80 GB** and **G4** (RTX PRO 6000 Blackwell). Open the one for your runtime and model and Run all: it
installs only that engine and serves only that model through one authenticated OpenAI-compatible API.

| Folder | What's in it |
|---|---|
| [`notebooks/l4-rtx4070ti/`](notebooks/l4-rtx4070ti/) | L4 (RTX 4070 Ti class, 24 GB): one folder per engine, one notebook per model |
| [`notebooks/a100/`](notebooks/a100/) | A100 80 GB (High-RAM) |
| [`notebooks/g4-rtx-pro6000/`](notebooks/g4-rtx-pro6000/) | G4 (RTX PRO 6000 Blackwell, 96 GB), including the G4-only engines (NInfer, Basalt, Strata NVFP4) |
| [`notebooks/cpu/prepare_cpu.ipynb`](notebooks/cpu/prepare_cpu.ipynb) | **Save GPU time:** on a cheap CPU runtime, download a model to Drive and build its engine for the GPU you will use |
| [`notebooks/other/unsloth/`](notebooks/other/unsloth/) | Official Unsloth Studio |

The full list of notebooks is in [`notebooks/README.md`](notebooks/README.md) (and the
[`llm_hub_colab.ipynb`](notebooks/llm_hub_colab.ipynb) index). Engine picks and workstation notes:
[`notebook_code/ENGINE_MATRIX.md`](notebook_code/ENGINE_MATRIX.md). Settings are
`settings/settings_<engine>.json`; weights and builds share Drive.

<!-- RESULTS:START -->

## Tested on Colab

Measured with the notebooks' hub on real Colab runtimes, one request at a time; the numbers live in
[`results/results.csv`](results/results.csv) and the images and videos in
[`results/samples/`](results/samples/). Sorted by runtime, then engine.

- **Prefill:** prompt tokens per second for a ~3.5K-token prompt (fresh, no prefix-cache reuse).
- **Time to first token:** for that prompt, on a loaded model.
- **Decode:** a 512-token reply at temperature 0.
- **Cold start:** the hub starting the engine, loading the model from Drive and answering a short prompt.
- *next run*: measured from the next test run on (earlier runs used a prompt too short for prefill).

### L4 (RTX 4070 Ti class, 24 GB)

| Engine | Model notebook | Prefill | Time to first token | Decode | Cold start |
|---|---|---|---|---|---|
| llama.cpp | `nemotron-3.5-bolt-74b-iq2` | next run | next run | 38 tok/s | 7.8 min |
| llama.cpp | `qwen3.8-27b-q4` | next run | next run | 15 tok/s | 5 min |
| SGLang | `qwen3.8-27b-w4a16-sglang` | next run | next run | 16 tok/s | 4.5 min |
| Strata | `flash-next-gyro-s` | next run | next run | 68 tok/s | 24.3 min |
| Strata | `qwen36-strata-q4` | next run | next run | 55 tok/s | 7.3 min |
| TensorFold | `qwen3.8-27b-exl3-2bpw` | next run | next run | 83 tok/s | 7.5 min |
| vLLM | `qwen3-14b-fp8` | next run | next run | 17 tok/s | 7.3 min |
| vLLM | `qwen3.8-27b-w4a16` | next run | next run | 16 tok/s | 14.8 min |

| Engine | Model notebook | Per image / 5 s clip (after the first) | Cold start | Sample |
|---|---|---|---|---|
| ComfyUI | `flux3-klein-comfy` | 8 s per image | 3.5 min | not saved |
| ComfyUI | `minimax-h3-l4-comfy` | 124 s per clip | not recorded | [l4_minimax-h3-l4-comfy.mp4](results/samples/l4_minimax-h3-l4-comfy.mp4) |
| ComfyUI | `qwen-image-2.1-turbo-comfy` | 8 s per image | not recorded | <a href="results/samples/l4_qwen-image-2.1-turbo-comfy.jpg"><img src="results/samples/l4_qwen-image-2.1-turbo-comfy.jpg" width="160"></a> |
| Diffusers | `qwen-image-2.1-nf4` | 37 s per image | 12.7 min | not saved |
| Diffusers | `sdxl-base` | 17 s per image | 6.1 min | not saved |
| Diffusers | `sdxl-turbo` | 3 s per image | 7.4 min | not saved |

### A100 80 GB (High-RAM)

| Engine | Model notebook | Prefill | Time to first token | Decode | Cold start |
|---|---|---|---|---|---|
| HyperQwen | `hyperqwen-qwen3.8-27b-batch` | 3,709 tok/s | 0.96 s | 66 tok/s | 9.8 min |
| llama.cpp | `nemotron-3.5-bolt-74b-iq2` | next run | next run | 79 tok/s | 7.4 min |
| llama.cpp | `qwen3.8-27b-q8` | 1,314 tok/s | 2.71 s | 39 tok/s | 6.2 min |
| llama.cpp | `sage-a1.5-q4` | 3,558 tok/s | 1.00 s | 134 tok/s | 10.7 min |
| SGLang | `qwen3.8-27b-w4a16-sglang` | 2,018 tok/s | 1.76 s | 74 tok/s | 17.6 min |
| Strata | `flash-next-strata-iq3-s` | next run | next run | 167 tok/s | 29.6 min |
| TensorFold | `qwen3.8-27b-exl3-2bpw` | next run | next run | 144 tok/s | 5.9 min |
| vLLM | `qwen3.8-27b-fp8` | 2,905 tok/s | 1.22 s | 48 tok/s | 21.8 min |

| Engine | Model notebook | Per image / 5 s clip (after the first) | Cold start | Sample |
|---|---|---|---|---|
| ComfyUI | `flux3-klein-comfy` | 3 s per image | 3.7 min | not saved |
| ComfyUI | `minimax-h3-comfy` | 35 s per clip | not recorded | [a100_minimax-h3-comfy.mp4](results/samples/a100_minimax-h3-comfy.mp4) |
| ComfyUI | `qwen-image-2.1-bf16-comfy` | 9 s per image | not recorded | <a href="results/samples/a100_qwen-image-2.1-bf16-comfy.jpg"><img src="results/samples/a100_qwen-image-2.1-bf16-comfy.jpg" width="160"></a> |
| ComfyUI | `qwen-image-2.1-turbo-bf16-comfy` | 4 s per image | not recorded | <a href="results/samples/a100_qwen-image-2.1-turbo-bf16-comfy.jpg"><img src="results/samples/a100_qwen-image-2.1-turbo-bf16-comfy.jpg" width="160"></a> |
| ComfyUI | `wan22-5b-comfy` | 61 s per clip | 6.4 min | [a100_wan22-5b-comfy.mp4](results/samples/a100_wan22-5b-comfy.mp4) |
| Diffusers | `qwen-image-2.1` | 16 s per image | 12.4 min | <a href="results/samples/a100_qwen-image-2.1.jpg"><img src="results/samples/a100_qwen-image-2.1.jpg" width="160"></a> |

### G4 (RTX PRO 6000 Blackwell, 96 GB)

| Engine | Model notebook | Prefill | Time to first token | Decode | Cold start |
|---|---|---|---|---|---|
| Basalt | `basalt-flash-next-nvfp4-dq6k` | next run | next run | 273 tok/s | not recorded |
| HyperQwen | `hyperqwen-qwen3.8-27b-dflash2` | next run | next run | 315 tok/s | 5.1 min |
| Maya | `glm-maya-s` | next run | next run | 77 tok/s | 25 min |
| NInfer | `ninfer-qwen3.8-27b-nvfp4` | next run | next run | 187 tok/s | 67 s |
| NInfer | `ninfer-qwen3.8-flash-next` | next run | next run | 66 tok/s | not recorded |
| SGLang | `qwen3.8-27b-w4a16-sglang` | next run | next run | 77 tok/s | not recorded |
| Strata NVFP4 | `strata-nvfp4-flash-next-gptq` | next run | next run | 187 tok/s | 17 min |

Tested and then dropped for a faster quant of the same model (kept in the CSV): L4 Strata `flash-next-strata-q2` (46 tok/s), L4 Strata `flash-next-swift-iq2-xs` (55 tok/s).

<!-- RESULTS:END -->

## Prepare on a CPU runtime, serve on a GPU runtime

Downloading a model and building an engine need no GPU, so do them on a **CPU** runtime with
[`notebooks/cpu/prepare_cpu.ipynb`](notebooks/cpu/prepare_cpu.ipynb): pick the GPU you will use
(L4 = RTX 4070 Ti class, A100 or G4 = RTX PRO 6000), the engine and the model. It downloads the model straight to Drive one file at a time
(resumable; Drive accepts about 750 GB of uploads per day) while it builds or caches the engine for that GPU:
wheels, HyperQwen's venv, llama.cpp / NInfer compiled for the GPU's architecture. The GPU notebook then only
copies them from Drive and loads the model. If Drive is not available (the mount fails, or you untick
`USE_DRIVE` in the first cell), a GPU notebook downloads the model straight to its own disk instead; nothing is
kept after the runtime stops. Without a browser:
`python -m notebook_code prepare --target g4 --engine ninfer --model recommended`.

## Google AI Pro / Colab Pro

Use a Pro runtime with an **L4** (RTX 4070 Ti class), **A100 80 GB**, or **G4** (RTX PRO 6000 Blackwell) GPU. The L4 uses the same Ada chip as an RTX 4070 Ti with 24 GB, at about 300 GB/s and 72 W, so it is much slower than a 4090. For the A100, pick **High-RAM** in Change runtime type: that gives the 80 GB GPU these notebooks are sized for.

## Code
`notebook_code/` is the Python package behind the hub notebooks (hub server, Drive model store, catalog, backends, tests). Each notebook carries it packed in a cell, so nothing is fetched from GitHub. Regenerate every notebook with `python3 -m notebook_code.make_engine_notebooks`.

## Credit and license
The Studio notebook is from [unslothai/unsloth](https://github.com/unslothai/unsloth); Unsloth Studio is licensed under AGPL-3.0.
llama.cpp is from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
TensorFold is from [ashhart/TensorFold](https://github.com/ashhart/TensorFold); Sparsa from [LyJonathan/Sparsa](https://github.com/LyJonathan/Sparsa); HyperQwen from [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen); NInfer from [Neroued/ninfer](https://github.com/Neroued/ninfer); Basalt from [jesdga95/basalt](https://github.com/jesdga95/basalt); the Strata NVFP4 fork from [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4); ComfyUI from [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI).
Diffusers / Comfy models keep their upstream licenses (Stability AI, Black Forest Labs, etc.); accept gated licenses on Hugging Face and set Colab secret `HF_TOKEN` when needed. Runtimes driven from the command line cannot read Colab Secrets; for those, put the token in a one-line file on your private Drive, `MyDrive/colab_model_server/access/hf_token.txt`, next to the other keys (a Colab secret still wins when present).

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
