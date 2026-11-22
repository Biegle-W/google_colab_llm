# Colab notebooks: runtime → engine → model

`<runtime>/<engine>/<model>.ipynb`. Open the notebook for your runtime and model in Colab, set the runtime
type it names, and Run all. It installs only that engine, serves only that model (one authenticated
OpenAI-compatible API), and keeps the weights and engine builds on Google Drive.

### L4 (RTX 4070 Ti class, 24 GB) — `l4-rtx4070ti/`

| Engine | Model notebook |
|---|---|
| **llama.cpp** (`llama/`) | [`inkling-2-mini-iq4`](l4-rtx4070ti/llama/inkling-2-mini-iq4.ipynb) |
| **SGLang** (`sglang/`) | [`qwen3.8-27b-w4a16-sglang`](l4-rtx4070ti/sglang/qwen3.8-27b-w4a16-sglang.ipynb) |
| **Diffusers** (`diffusers/`) | [`qwen-image-2.1-nf4`](l4-rtx4070ti/diffusers/qwen-image-2.1-nf4.ipynb), [`sdxl-turbo`](l4-rtx4070ti/diffusers/sdxl-turbo.ipynb), [`sdxl-base`](l4-rtx4070ti/diffusers/sdxl-base.ipynb) |
| **Strata** (`strata/`) | [`flash-next-gyro-s`](l4-rtx4070ti/strata/flash-next-gyro-s.ipynb) |
| **TensorFold** (`tensorfold/`) | [`qwen3.8-27b-exl3-2bpw`](l4-rtx4070ti/tensorfold/qwen3.8-27b-exl3-2bpw.ipynb) |
| **ComfyUI** (`comfyui/`) | [`qwen-image-2.1-turbo-comfy`](l4-rtx4070ti/comfyui/qwen-image-2.1-turbo-comfy.ipynb), [`flux3-klein-comfy`](l4-rtx4070ti/comfyui/flux3-klein-comfy.ipynb), [`minimax-h3-image-l4-comfy`](l4-rtx4070ti/comfyui/minimax-h3-image-l4-comfy.ipynb), [`minimax-h3-l4-comfy`](l4-rtx4070ti/comfyui/minimax-h3-l4-comfy.ipynb) |

### A100 80 GB (High-RAM) — `a100/`

| Engine | Model notebook |
|---|---|
| **llama.cpp** (`llama/`) | [`inkling-2-mini-q8`](a100/llama/inkling-2-mini-q8.ipynb) |
| **SGLang** (`sglang/`) | [`qwen3.8-27b-w4a16-sglang`](a100/sglang/qwen3.8-27b-w4a16-sglang.ipynb) |
| **Diffusers** (`diffusers/`) | [`flux-schnell`](a100/diffusers/flux-schnell.ipynb) |
| **Strata** (`strata/`) | [`flash-next-strata-iq3-s`](a100/strata/flash-next-strata-iq3-s.ipynb), [`deepseek-strata-ist-2.00`](a100/strata/deepseek-strata-ist-2.00.ipynb), [`hy4-flash-sq-128`](a100/strata/hy4-flash-sq-128.ipynb) |
| **Maya** (`maya/`) | [`glm-maya-s`](a100/maya/glm-maya-s.ipynb), [`hy4-flash-maya-l`](a100/maya/hy4-flash-maya-l.ipynb) |
| **TensorFold** (`tensorfold/`) | [`qwen3.8-27b-exl3-2bpw`](a100/tensorfold/qwen3.8-27b-exl3-2bpw.ipynb) |
| **ComfyUI** (`comfyui/`) | [`qwen-image-2.1-turbo-bf16-comfy`](a100/comfyui/qwen-image-2.1-turbo-bf16-comfy.ipynb), [`qwen-image-2.1-bf16-comfy`](a100/comfyui/qwen-image-2.1-bf16-comfy.ipynb), [`flux3-klein-comfy`](a100/comfyui/flux3-klein-comfy.ipynb), [`minimax-h3-image-comfy`](a100/comfyui/minimax-h3-image-comfy.ipynb), [`wan22-5b-comfy`](a100/comfyui/wan22-5b-comfy.ipynb), [`minimax-h3-comfy`](a100/comfyui/minimax-h3-comfy.ipynb) |

### G4 (RTX PRO 6000 Blackwell, 96 GB) — `g4-rtx-pro6000/`

| Engine | Model notebook |
|---|---|
| **llama.cpp** (`llama/`) | [`inkling-2-mini-q8`](g4-rtx-pro6000/llama/inkling-2-mini-q8.ipynb) |
| **SGLang** (`sglang/`) | [`qwen3.8-27b-w4a16-sglang`](g4-rtx-pro6000/sglang/qwen3.8-27b-w4a16-sglang.ipynb) |
| **Diffusers** (`diffusers/`) | [`flux-schnell`](g4-rtx-pro6000/diffusers/flux-schnell.ipynb) |
| **Strata** (`strata/`) | [`deepseek-strata-ist-1.75`](g4-rtx-pro6000/strata/deepseek-strata-ist-1.75.ipynb), [`hy4-flash-sq-128`](g4-rtx-pro6000/strata/hy4-flash-sq-128.ipynb) |
| **Maya** (`maya/`) | [`glm-maya-s`](g4-rtx-pro6000/maya/glm-maya-s.ipynb), [`hy4-flash-maya-l`](g4-rtx-pro6000/maya/hy4-flash-maya-l.ipynb) |
| **ComfyUI** (`comfyui/`) | [`qwen-image-2.1-turbo-bf16-comfy`](g4-rtx-pro6000/comfyui/qwen-image-2.1-turbo-bf16-comfy.ipynb), [`qwen-image-2.1-bf16-comfy`](g4-rtx-pro6000/comfyui/qwen-image-2.1-bf16-comfy.ipynb), [`flux3-klein-comfy`](g4-rtx-pro6000/comfyui/flux3-klein-comfy.ipynb), [`minimax-h3-image-g4-comfy`](g4-rtx-pro6000/comfyui/minimax-h3-image-g4-comfy.ipynb), [`wan22-5b-comfy`](g4-rtx-pro6000/comfyui/wan22-5b-comfy.ipynb), [`minimax-h3-g4-comfy`](g4-rtx-pro6000/comfyui/minimax-h3-g4-comfy.ipynb) |
| **HyperQwen** (`hyperqwen/`) | [`hyperqwen-qwen3.8-27b-dflash2`](g4-rtx-pro6000/hyperqwen/hyperqwen-qwen3.8-27b-dflash2.ipynb) |
| **Basalt** (`basalt/`) | [`basalt-flash-next-nvfp4-dq6k`](g4-rtx-pro6000/basalt/basalt-flash-next-nvfp4-dq6k.ipynb) |

## Other

- [`cpu/prepare_cpu.ipynb`](cpu/prepare_cpu.ipynb): on a cheap CPU runtime, download a model to Drive and build
  its engine for the GPU you will use, so the GPU notebook only restores them.
- [`other/unsloth/`](other/unsloth/): official Unsloth Studio.

Regenerate every notebook from the repo root (the hub code is packed into each one):

```bash
python3 -m notebook_code.make_engine_notebooks
```
