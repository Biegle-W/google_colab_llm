# Google Colab LLM + diffusion notebooks

Notebooks for running **LLM and text-to-image** tooling on Colab GPUs (including **Google AI Pro / Colab Pro** L4, A100, and G4 runtimes). Models download once to **Google Drive**, then restore into later sessions. Set Runtime → Change runtime type to the hardware named for the notebook, then Run all.

| Notebook | What it does |
|---|---|
| `llm_hub_colab.ipynb` | **Main hub:** OpenAI-compatible chat + images for **Qwen3.8-27B**, **Flash-Next**, **GLM-5.3-Flash**, **DeepSeek-V4.1-Flash-1017**, **Qwen-Image-2.1** (MiniMax-H3 notes); engines auto-picked per GPU (llama.cpp / vLLM / SGLang / Diffusers); Drive-backed; see [`colab_llm_hub/ENGINE_MATRIX.md`](colab_llm_hub/ENGINE_MATRIX.md) |
| `Unsloth_Studio_Colab.ipynb` | Official Unsloth Studio (web UI for training and chat) |
| `llama_cpp_t4.ipynb` | Build llama.cpp with CUDA, run Qwen3.8-27B Q4_K_M (partial GPU offload on a T4), chat via llama-server |
| `llama_cpp_gpu.ipynb` | llama.cpp on an L4, A100 or G4: detects the GPU, builds for it, and picks a Qwen3.8-27B GGUF that fits its memory |
| `vllm_gpu.ipynb` | vLLM on an L4, A100 or G4: picks Qwen3.8-27B FP8 (A100, G4) or Qwen3-14B FP8 (L4), with tool calling |
| `vllm_tpu.ipynb` | vLLM on a v5e-1 TPU; checks whether the model fits in 16 GB HBM and falls back to Qwen3-4B (the 27B does not fit) |

## Google AI Pro / Colab Pro

Use a Pro runtime with an **L4, A100, or G4** GPU for the hub notebook. That unlocks the larger catalog LLMs and FLUX / SDXL image models. A free T4 is fine for `llama_cpp_t4.ipynb` and the small `sd15` Diffusers entry, but most hub LLMs will not fit.

## Code
`colab_llm_hub/` is the Python package behind `llm_hub_colab.ipynb` (hub server, Drive model store, catalog, llama.cpp / vLLM / Diffusers backends, tests). The notebook fetches it from this repository each run.

## Credit and license
The Studio notebook is from [unslothai/unsloth](https://github.com/unslothai/unsloth); Unsloth Studio is licensed under AGPL-3.0.
llama.cpp is from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
Diffusers models keep their upstream licenses (Stability AI, Black Forest Labs, etc.); accept gated licenses on Hugging Face and set Colab secret `HF_TOKEN` when needed.

## Model and build storage
Notebooks mount Google Drive and keep **all durable state** under `MyDrive/llm_models` so later sessions restore instead of re-downloading:

| Path | Contents |
|---|---|
| `models/<id>/` | Weights (GGUF, safetensors, Diffusers trees), marked complete only after a full copy |
| `builds/` | llama.cpp CUDA tarballs; vLLM / SGLang / Diffusers / hub_deps wheels; cached `hub_pkg/`; `tools/cloudflared` |
| `config/settings.json` | Prefetch list, engines, context overrides, tunnel/port |
| `hf_cache/` | Hugging Face / Transformers / Diffusers cache (`HF_HOME`) |
| `hub/api_key`, `hub/endpoint.json` | Stable Bearer token + last public URL |
| `logs/` | `install.log`, `hub.log`, `cloudflared.log`, engine `hub_logs/` |
| `outputs/images/`, `outputs/chat/`, `outputs/status/` | Generation results and status snapshots |

A model is downloaded from Hugging Face once, saved to Drive, and copied to local disk in later sessions (engines load faster from local disk than through the Drive FUSE mount). The first run asks you to authorize Drive access. Set `REFRESH_WHEELS` / `REBUILD_LLAMA` in Drive settings to force updates.

## Remote access
`llm_hub_colab.ipynb`, `llama_cpp_*.ipynb`, `vllm_*.ipynb` can open a free Cloudflare tunnel to their OpenAI-compatible server, protected by a random (or Drive-persisted) API key printed in the notebook. Both also run DeepSeek Harness (dsh) as an agent front end behind a tokenized link where applicable. The URL lasts for one quick-tunnel session unless you configure a named tunnel.

Builds are cached too: the first run builds/downloads and saves them; later runs restore them. Set `REBUILD_LLAMA` or `REFRESH_WHEELS` in the notebook to update. A cache is tied to the Python version (and, for llama.cpp, the GPU arch), so if Colab changes its image the notebook rebuilds automatically.
