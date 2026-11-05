# Google Colab LLM + diffusion notebooks

Notebooks for running **LLM and text-to-image** tooling on Colab GPUs (including **Google AI Pro / Colab Pro** L4, A100, and G4 runtimes). Models download once to **Google Drive**, then restore into later sessions. Set Runtime → Change runtime type to the hardware named for the notebook, then Run all.

| Notebook | What it does |
|---|---|
| `llm_hub_colab.ipynb` | **Index:** links to one hub notebook per engine |
| `strata_hub_colab.ipynb` | **Strata only** — Flash-Next (ISTA/Gyro/Swift) + Qwen3.6 + DeepSeek |
| `maya_hub_colab.ipynb` | **Maya only** — GLM-5.3-Flash + DeepSeek-V4.1 (maya-ds4) |
| `tensorfold_hub_colab.ipynb` | **TensorFold only** — Qwen3.8-27B EXL3 2.0bpw + DFlash2 |
| `sparsa_hub_colab.ipynb` | **Sparsa only** — MiniMax-M3.1-Flash (IQ2_XS-D3 / IQ2_KT) |
| `comfyui_hub_colab.ipynb` | **ComfyUI only** — Qwen-Image-2.1 Turbo, FLUX.3 Klein, video prefetch |
| `llama_hub_colab.ipynb` | **llama.cpp only** — Bolt IQ2, Sage, Ornith, Qwen / MoE GGUF fallbacks |
| `vllm_hub_colab.ipynb` | **vLLM only** — Qwen3.8-27B W4A16/FP8/BF16, Sage, Bolt W4A16 |
| `sglang_hub_colab.ipynb` | **SGLang only** — Qwen3.8-27B W4A16 AutoRound |
| `diffusers_hub_colab.ipynb` | **Diffusers only** — Qwen-Image-2.1, FLUX, SDXL (+ MiniMax-H3 prefetch notes) |
| `Unsloth_Studio_Colab.ipynb` | Official Unsloth Studio (web UI for training and chat) |
| `llama_cpp_t4.ipynb` | Older single-model llama.cpp T4 notebook |
| `llama_cpp_gpu.ipynb` | Older single-model llama.cpp L4/A100/G4 notebook |
| `vllm_gpu.ipynb` | Older single-model vLLM GPU notebook |
| `vllm_tpu.ipynb` | vLLM on a v5e-1 TPU; checks whether the model fits in 16 GB HBM and falls back to Qwen3-4B (the 27B does not fit) |

Engine picks and workstation notes: [`colab_llm_hub/ENGINE_MATRIX.md`](colab_llm_hub/ENGINE_MATRIX.md). Each `*_hub_colab.ipynb` installs **only its engine** and lists **only that engine’s catalog models**. Settings are `config/settings_<engine>.json`; weights/builds still share Drive.

## Google AI Pro / Colab Pro

Use a Pro runtime with an **L4, A100, or G4** GPU for the hub notebooks. That unlocks the larger catalog LLMs and FLUX / SDXL / ComfyUI image models. A free T4 is fine for `llama_cpp_t4.ipynb` and Diffusers NF4 / `sd15` / ComfyUI Turbo, but most hub LLMs will not fit.

## Code
`colab_llm_hub/` is the Python package behind the `*_hub_colab.ipynb` notebooks (hub server, Drive model store, catalog, backends, tests). Each notebook fetches it from this repository each run. Regenerate engine notebooks with `python3 -m colab_llm_hub.make_engine_notebooks`.

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
| `hub/api_key`, `hub/endpoint.json` | Stable Bearer token + last public URL |
| `logs/` | `install.log`, `hub.log`, `cloudflared.log`, engine `hub_logs/` |
| `outputs/images/`, `outputs/chat/`, `outputs/status/` | Generation results and status snapshots |

A model is downloaded from Hugging Face once, saved to Drive, and copied to local disk in later sessions (engines load faster from local disk than through the Drive FUSE mount). The first run asks you to authorize Drive access. Set `REFRESH_WHEELS` / `REBUILD_LLAMA` in Drive settings to force updates.

## Remote access
The `*_hub_colab.ipynb` notebooks (and older `llama_cpp_*.ipynb` / `vllm_*.ipynb`) can open a free Cloudflare tunnel to their OpenAI-compatible server, protected by a random (or Drive-persisted) API key printed in the notebook. They can also print a DeepSeek Harness (dsh) provider block. The URL lasts for one quick-tunnel session unless you configure a named tunnel.

Builds are cached too: the first run builds/downloads and saves them; later runs restore them. Set `REBUILD_LLAMA` or `REFRESH_WHEELS` in the notebook to update. A cache is tied to the Python version (and, for llama.cpp, the GPU arch), so if Colab changes its image the notebook rebuilds automatically.
