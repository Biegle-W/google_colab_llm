# Google Colab LLM + diffusion notebooks

Notebooks for running **LLM and text-to-image** tooling on Colab GPUs (including **Google AI Pro / Colab Pro** L4, A100, and G4 runtimes). Models download once to **Google Drive**, then restore into later sessions. Set Runtime → Change runtime type to the hardware named for the notebook, then Run all.

Notebooks are grouped by engine under [`notebooks/`](notebooks/).

| Notebook | What it does |
|---|---|
| [`notebooks/llm_hub_colab.ipynb`](notebooks/llm_hub_colab.ipynb) | **Index:** links to one hub notebook per engine |
| [`notebooks/strata/`](notebooks/strata/) | **Strata** — Flash-Next (ISTA/Gyro/Swift) + Qwen3.6 + DeepSeek |
| [`notebooks/maya/`](notebooks/maya/) | **Maya** — GLM-5.3-Flash + DeepSeek-V4.1 (maya-ds4) |
| [`notebooks/tensorfold/`](notebooks/tensorfold/) | **TensorFold** — Qwen3.8-27B EXL3 2.0bpw + DFlash2 |
| [`notebooks/sparsa/`](notebooks/sparsa/) | **Sparsa** — MiniMax-M3.1-Flash (IQ2_XS-D3 / IQ2_KT) |
| [`notebooks/comfyui/`](notebooks/comfyui/) | **ComfyUI** — Qwen-Image-2.1 Turbo, FLUX.3 Klein, video prefetch |
| [`notebooks/llama/`](notebooks/llama/) | **llama.cpp** hub + older `llama_cpp_*.ipynb` |
| [`notebooks/vllm/`](notebooks/vllm/) | **vLLM** hub + older `vllm_gpu` / `vllm_tpu` |
| [`notebooks/sglang/`](notebooks/sglang/) | **SGLang** — Qwen3.8-27B W4A16 AutoRound |
| [`notebooks/diffusers/`](notebooks/diffusers/) | **Diffusers** — Qwen-Image-2.1, FLUX, SDXL |
| [`notebooks/unsloth/`](notebooks/unsloth/) | Official Unsloth Studio |

Engine picks and workstation notes: [`colab_llm_hub/ENGINE_MATRIX.md`](colab_llm_hub/ENGINE_MATRIX.md). Each `*_hub_colab.ipynb` installs **only its engine** and lists **only that engine’s catalog models**. Settings are `config/settings_<engine>.json`; weights/builds still share Drive.

## Google AI Pro / Colab Pro

Use a Pro runtime with an **L4, A100, or G4** GPU for the hub notebooks. That unlocks the larger catalog LLMs and FLUX / SDXL / ComfyUI image models. A free T4 is fine for `notebooks/llama/llama_cpp_t4.ipynb` and Diffusers NF4 / `sd15` / ComfyUI Turbo, but most hub LLMs will not fit.

## Code
`colab_llm_hub/` is the Python package behind the hub notebooks (hub server, Drive model store, catalog, backends, tests). Each notebook fetches it from this repository each run. Regenerate engine notebooks with `python3 -m colab_llm_hub.make_engine_notebooks`.

## Credit and license
The Studio notebook is from [unslothai/unsloth](https://github.com/unslothai/unsloth); Unsloth Studio is licensed under AGPL-3.0.
llama.cpp is from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
TensorFold is from [ashhart/TensorFold](https://github.com/ashhart/TensorFold); Sparsa from [LyJonathan/Sparsa](https://github.com/LyJonathan/Sparsa); ComfyUI from [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI).
Diffusers / Comfy models keep their upstream licenses (Stability AI, Black Forest Labs, etc.); accept gated licenses on Hugging Face and set Colab secret `HF_TOKEN` when needed.

## Model and build storage
Notebooks mount Google Drive and keep **all durable state** under `MyDrive/llm_models` so later sessions restore instead of re-downloading:

| Path | Contents |
|---|---|
| `models/<id>/` | Weights (GGUF, safetensors, Diffusers / Comfy trees), marked complete only after a full copy |
| `builds/` | llama.cpp tarballs; wheels; TensorFold / Sparsa / ComfyUI / Strata / Maya trees; `hub_pkg/`; `tools/cloudflared` |
| `config/settings_<engine>.json` | Per-engine prefetch, ctx overrides, tunnel/port |
| `hf_cache/` | Hugging Face / Transformers / Diffusers cache (`HF_HOME`) |
| `hub/api_key`, `hub/tailscale_authkey`, `hub/endpoint.json` | Private Drive secrets + last public URL (not in git) |
| `logs/` | `install.log`, `hub.log`, `cloudflared.log`, engine `hub_logs/` |
| `outputs/images/`, `outputs/chat/`, `outputs/status/` | Generation results and status snapshots |

A model is downloaded from Hugging Face once, saved to Drive, and copied to local disk in later sessions (engines load faster from local disk than through the Drive FUSE mount). The first run asks you to authorize Drive access. Set `REFRESH_WHEELS` / `REBUILD_LLAMA` in Drive settings to force updates.

## Remote access
Hub notebooks default to **Tailscale**. On first run you paste a Tailscale auth key (reusable + ephemeral recommended); it is saved only on **your private Google Drive** at `MyDrive/llm_models/hub/tailscale_authkey` (gitignored; never committed). Later Colab sessions reuse that Drive file. Optional Colab secret `TAILSCALE_AUTHKEY` if you press Enter on an empty prompt. Cloudflare `quick` / `named` remain available via `TUNNEL` in settings. The hub API key is also Drive-only (`hub/api_key`).

Builds are cached too: the first run builds/downloads and saves them; later runs restore them. Set `REBUILD_LLAMA` or `REFRESH_WHEELS` in the notebook to update. A cache is tied to the Python version (and, for llama.cpp, the GPU arch), so if Colab changes its image the notebook rebuilds automatically.
