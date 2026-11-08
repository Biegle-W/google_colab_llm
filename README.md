# Google Colab LLM + diffusion notebooks

Notebooks for running **LLM and text-to-image** tooling on Colab GPUs (including **Google AI Pro / Colab Pro** L4, A100, and G4 runtimes). Models download once to **Google Drive**, then restore into later sessions. Set Runtime → Change runtime type to the hardware named for the notebook, then Run all.

Notebooks are grouped by engine under [`notebooks/`](notebooks/).

| Notebook | What it does |
|---|---|
| [`notebooks/llm_hub_colab.ipynb`](notebooks/llm_hub_colab.ipynb) | **Index:** links to the GPU notebook of each engine |
| [`notebooks/strata/`](notebooks/strata/) | **Strata** — Flash-Next (ISTA/Gyro/Swift) + Qwen3.6 + DeepSeek |
| [`notebooks/maya/`](notebooks/maya/) | **Maya** — GLM-5.3-Flash + DeepSeek-V4.1 (maya-ds4) |
| [`notebooks/tensorfold/`](notebooks/tensorfold/) | **TensorFold** — Qwen3.8-27B EXL3 2.0bpw + DFlash2 |
| [`notebooks/sparsa/`](notebooks/sparsa/) | **Sparsa** — MiniMax-M3.1-Flash (IQ2_XS-D3 / IQ2_KT) |
| [`notebooks/hyperqwen/`](notebooks/hyperqwen/) | **HyperQwen** — Qwen3.8-27B W4A16 on patched vLLM (MTP / DFlash2 / batch); A100 batch only, G4 all |
| [`notebooks/ninfer/`](notebooks/ninfer/) | **NInfer** — native C++/CUDA Qwen3.8-27B, Qwen3.6-35B-A3B, Flash-Next; **G4 only** |
| [`notebooks/basalt/`](notebooks/basalt/) | **Basalt** — Strata fork for Blackwell, Qwen3.8-Flash-Next NVFP4-DQ6_K; **G4 only** |
| [`notebooks/strata_nvfp4/`](notebooks/strata_nvfp4/) | **Strata NVFP4 fork** — Qwen3.8-Flash-Next NVFP4 GPTQ; **G4 only** |
| [`notebooks/comfyui/`](notebooks/comfyui/) | **ComfyUI** — Qwen-Image-2.1 Turbo, FLUX.3 Klein, video prefetch |
| [`notebooks/llama/`](notebooks/llama/) | **llama.cpp** — `llama_gpu.ipynb` (L4/A100/G4) + `llama_cpp_t4.ipynb` (T4) |
| [`notebooks/vllm/`](notebooks/vllm/) | **vLLM** — `vllm_gpu.ipynb` (L4/A100/G4) + `vllm_tpu.ipynb` (TPU) |
| [`notebooks/sglang/`](notebooks/sglang/) | **SGLang** — Qwen3.8-27B W4A16 AutoRound |
| [`notebooks/diffusers/`](notebooks/diffusers/) | **Diffusers** — Qwen-Image-2.1, FLUX, SDXL |
| [`notebooks/prepare/prepare_cpu.ipynb`](notebooks/prepare/prepare_cpu.ipynb) | **Save GPU time:** on a cheap CPU runtime, download a model to Drive and build its engine for the GPU you will use |
| [`notebooks/unsloth/`](notebooks/unsloth/) | Official Unsloth Studio |

Engine picks and workstation notes: [`notebook_code/ENGINE_MATRIX.md`](notebook_code/ENGINE_MATRIX.md). Each `<engine>_gpu.ipynb` runs the hub, installs **only its engine** and lists **only that engine’s catalog models**. Settings are `config/settings_<engine>.json`; weights/builds still share Drive.

## Prepare on a CPU runtime, serve on a GPU runtime

Downloading a model and building an engine need no GPU, so do them on a **CPU** runtime with
[`notebooks/prepare/prepare_cpu.ipynb`](notebooks/prepare/prepare_cpu.ipynb): pick the GPU you will use
(L4, A100 or G4), the engine and the model. It downloads the model straight to Drive one file at a time
(resumable; Drive accepts about 750 GB of uploads per day) while it builds or caches the engine for that GPU:
wheels, HyperQwen's venv, llama.cpp / NInfer compiled for the GPU's architecture. The GPU notebook then only
copies them from Drive and loads the model. Without a browser:
`python -m notebook_code prepare --target g4 --engine ninfer --model recommended`.

## Google AI Pro / Colab Pro

Use a Pro runtime with an **L4, A100, or G4** GPU for the GPU notebooks. That unlocks the larger catalog LLMs and FLUX / SDXL / ComfyUI image models. A free T4 is fine for `notebooks/llama/llama_cpp_t4.ipynb` and Diffusers NF4 / `sd15` / ComfyUI Turbo, but most catalog LLMs will not fit.

## Code
`notebook_code/` is the Python package behind the hub notebooks (hub server, Drive model store, catalog, backends, tests). Each notebook fetches it from this repository each run. Regenerate engine notebooks with `python3 -m notebook_code.make_engine_notebooks`.

## Credit and license
The Studio notebook is from [unslothai/unsloth](https://github.com/unslothai/unsloth); Unsloth Studio is licensed under AGPL-3.0.
llama.cpp is from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
TensorFold is from [ashhart/TensorFold](https://github.com/ashhart/TensorFold); Sparsa from [LyJonathan/Sparsa](https://github.com/LyJonathan/Sparsa); HyperQwen from [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen); NInfer from [Neroued/ninfer](https://github.com/Neroued/ninfer); Basalt from [jesdga95/basalt](https://github.com/jesdga95/basalt); the Strata NVFP4 fork from [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4); ComfyUI from [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI).
Diffusers / Comfy models keep their upstream licenses (Stability AI, Black Forest Labs, etc.); accept gated licenses on Hugging Face and set Colab secret `HF_TOKEN` when needed.

## Model and build storage
Notebooks mount Google Drive and keep **all durable state** under `MyDrive/llm_models` so later sessions restore instead of re-downloading:

| Path | Contents |
|---|---|
| `models/<id>/` | Weights (GGUF, safetensors, Diffusers / Comfy trees), marked complete only after a full copy |
| `builds/` | llama.cpp tarballs; wheels; TensorFold / Sparsa / ComfyUI / Strata / Maya trees; HyperQwen venv and NInfer build tarballs; `tools/tailscale`, `tools/cloudflared` |
| `config/settings_<engine>.json` | Per-engine selected model, ctx overrides, Tailscale hostname/port |
| `hf_cache/` | Small Hugging Face files only (`HF_HOME`: login token, tokenizer / config lookups); model weights go to `models/` |
| `hub/api_key.txt`, `hub/tailscale_authkey.txt`, `hub/cloudflare_tunnel_token.txt`, `hub/endpoint.json` | Private Drive secrets + last public URL (not in git) |
| `logs/` | `install.log`, `hub.log`, `tailscaled.log`, engine `hub_logs/` |
| `outputs/images/`, `outputs/chat/`, `outputs/status/` | Generation results and status snapshots |

A model is downloaded from Hugging Face straight to Drive, once. A model that fits in VRAM + RAM is then read directly from Drive every session (llama.cpp reads it with `--no-mmap` and vLLM with `--safetensors-load-strategy eager`, so the mount is read sequentially). A model that streams weights from SSD during inference (the DeepSeek Engram packs, a GGUF larger than memory) gets its streamed files copied from Drive to the runtime's local disk every session, with the rest read from Drive. Measured on Colab: the first read from Drive runs at about 77 MB/s (23.7 GB in 5 min), and Colab's Drive client caches what it reads on the runtime's local disk (22.6 GB of that 23.7 GB file), so a Drive-read model still occupies local disk and fit checks count it. The first run asks you to authorize Drive access. Set `REFRESH_WHEELS` / `REBUILD_LLAMA` in Drive settings to force updates.

## Remote access
Cell 6 asks how your machine connects (type 1, 2 or 3; Enter keeps the last choice), so nothing in the notebook is edited:

| Choice | Setup | URL | Who can reach it |
|---|---|---|---|
| 1. Tailscale (default) | Tailscale account + auth key | Stable `https://colab-<engine>.<tailnet>.ts.net` | Only your tailnet |
| 2. Cloudflare quick tunnel | None (as in Unsloth Studio's notebook) | New `https://<words>.trycloudflare.com` every session | Anyone with the URL; the API key protects the hub |
| 3. Cloudflare named tunnel | A tunnel in your Cloudflare dashboard + its token | Your stable hostname | Public; the API key protects the hub |

Keys: add `TAILSCALE_AUTHKEY` (and `HF_TOKEN`, `CLOUDFLARE_TUNNEL_TOKEN`) once in Colab's Secrets panel (key icon, left sidebar) and no notebook asks for them. Otherwise cell 6 asks in a hidden input box and keeps the key on **your private Google Drive** (`MyDrive/llm_models/hub/tailscale_authkey.txt`, which the Drive app on a phone opens); type `1 new` at the tunnel prompt to replace it. The hub API key is also Drive-only (`hub/api_key.txt`). Nothing secret is ever written into a notebook or this repository.

The Cloudflare link is printed only after the tunnel has registered and the hub answers through it. Your harness (dsh or any OpenAI client) runs on your own machine, never in a notebook. The T4 / TPU notebooks offer `tailscale` and `cloudflare` too.

Fit checks use the GPU's VRAM, system RAM (for models that spill into it) and the local disk (streamed files, plus the Drive client's cache of everything read from Drive); the model table says why a model is excluded. Streamed files go to the larger of `/content` and `/mnt/local-scratch` (the A100 80 GB's extra ~350 GB NVMe disk).

Each notebook downloads and serves **one model**, picked in its `MODEL` dropdown (`recommended` = the first recommended model that fits the GPU). With `SERVE_ONLY_SELECTED` (on by default) the hub serves only that model, so a harness request for another model cannot start a large download.

Builds are cached too: the first run builds/downloads and saves them; later runs restore them. Set `REBUILD_LLAMA` or `REFRESH_WHEELS` in the notebook to update. A cache is tied to the Python version (and, for llama.cpp, the GPU arch), so if Colab changes its image the notebook rebuilds automatically.
