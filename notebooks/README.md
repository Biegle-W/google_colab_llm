# Colab notebooks: runtime → engine → model

`<runtime>/<engine>/<model>.ipynb`. Open the notebook for your runtime and model in Colab, set the runtime
type it names, and Run all. It installs only that engine, serves only that model (one authenticated
OpenAI-compatible API), and keeps the weights and engine builds on Google Drive.

### L4 (RTX 4070 Ti class, 24 GB) — `l4-rtx4070ti/`

| Engine | Model notebook |
|---|---|
| **llama.cpp** (`llama/`) | [`qwen3.8-27b-q4`](l4-rtx4070ti/llama/qwen3.8-27b-q4.ipynb), [`nemotron-3.5-bolt-74b-iq2`](l4-rtx4070ti/llama/nemotron-3.5-bolt-74b-iq2.ipynb) |
| **vLLM** (`vllm/`) | [`qwen3.8-27b-w4a16`](l4-rtx4070ti/vllm/qwen3.8-27b-w4a16.ipynb), [`qwen3-14b-fp8`](l4-rtx4070ti/vllm/qwen3-14b-fp8.ipynb) |
| **SGLang** (`sglang/`) | [`qwen3.8-27b-w4a16-sglang`](l4-rtx4070ti/sglang/qwen3.8-27b-w4a16-sglang.ipynb) |
| **Diffusers** (`diffusers/`) | [`qwen-image-2.1-nf4`](l4-rtx4070ti/diffusers/qwen-image-2.1-nf4.ipynb), [`sdxl-turbo`](l4-rtx4070ti/diffusers/sdxl-turbo.ipynb), [`sdxl-base`](l4-rtx4070ti/diffusers/sdxl-base.ipynb) |
| **Strata** (`strata/`) | [`flash-next-gyro-s`](l4-rtx4070ti/strata/flash-next-gyro-s.ipynb), [`qwen36-strata-q4`](l4-rtx4070ti/strata/qwen36-strata-q4.ipynb) |
| **TensorFold** (`tensorfold/`) | [`qwen3.8-27b-exl3-2bpw`](l4-rtx4070ti/tensorfold/qwen3.8-27b-exl3-2bpw.ipynb) |
| **ComfyUI** (`comfyui/`) | [`qwen-image-2.1-turbo-comfy`](l4-rtx4070ti/comfyui/qwen-image-2.1-turbo-comfy.ipynb), [`flux3-klein-comfy`](l4-rtx4070ti/comfyui/flux3-klein-comfy.ipynb), [`minimax-h3-l4-comfy`](l4-rtx4070ti/comfyui/minimax-h3-l4-comfy.ipynb) |

### A100 80 GB (High-RAM) — `a100/`

| Engine | Model notebook |
|---|---|
| **llama.cpp** (`llama/`) | [`qwen3.8-27b-q8`](a100/llama/qwen3.8-27b-q8.ipynb), [`nemotron-3.5-bolt-74b-iq2`](a100/llama/nemotron-3.5-bolt-74b-iq2.ipynb), [`sage-a1.5-q4`](a100/llama/sage-a1.5-q4.ipynb), [`flash-next-q3`](a100/llama/flash-next-q3.ipynb) |
| **vLLM** (`vllm/`) | [`qwen3.8-27b-fp8`](a100/vllm/qwen3.8-27b-fp8.ipynb), [`sage-a1.5-fp8`](a100/vllm/sage-a1.5-fp8.ipynb), [`nemotron-3.5-bolt-74b-w4a16`](a100/vllm/nemotron-3.5-bolt-74b-w4a16.ipynb) |
| **SGLang** (`sglang/`) | [`qwen3.8-27b-w4a16-sglang`](a100/sglang/qwen3.8-27b-w4a16-sglang.ipynb) |
| **Diffusers** (`diffusers/`) | [`qwen-image-2.1`](a100/diffusers/qwen-image-2.1.ipynb), [`flux-schnell`](a100/diffusers/flux-schnell.ipynb) |
| **Strata** (`strata/`) | [`flash-next-strata-iq3-s`](a100/strata/flash-next-strata-iq3-s.ipynb), [`qwen36-strata-q6`](a100/strata/qwen36-strata-q6.ipynb) |
| **Maya** (`maya/`) | [`glm-maya-s`](a100/maya/glm-maya-s.ipynb) |
| **TensorFold** (`tensorfold/`) | [`qwen3.8-27b-exl3-2bpw`](a100/tensorfold/qwen3.8-27b-exl3-2bpw.ipynb) |
| **ComfyUI** (`comfyui/`) | [`qwen-image-2.1-turbo-bf16-comfy`](a100/comfyui/qwen-image-2.1-turbo-bf16-comfy.ipynb), [`qwen-image-2.1-bf16-comfy`](a100/comfyui/qwen-image-2.1-bf16-comfy.ipynb), [`flux3-klein-comfy`](a100/comfyui/flux3-klein-comfy.ipynb), [`wan22-5b-comfy`](a100/comfyui/wan22-5b-comfy.ipynb), [`minimax-h3-comfy`](a100/comfyui/minimax-h3-comfy.ipynb) |
| **Sparsa** (`sparsa/`) | [`minimax-m3.1-flash-iq2-kt`](a100/sparsa/minimax-m3.1-flash-iq2-kt.ipynb) |
| **HyperQwen** (`hyperqwen/`) | [`hyperqwen-qwen3.8-27b-batch`](a100/hyperqwen/hyperqwen-qwen3.8-27b-batch.ipynb) |

### G4 (RTX PRO 6000 Blackwell, 96 GB) — `g4-rtx-pro6000/`

| Engine | Model notebook |
|---|---|
| **llama.cpp** (`llama/`) | [`nemotron-3.5-bolt-74b-iq2`](g4-rtx-pro6000/llama/nemotron-3.5-bolt-74b-iq2.ipynb), [`flash-next-q3`](g4-rtx-pro6000/llama/flash-next-q3.ipynb), [`ornith-1.5-122b-q4`](g4-rtx-pro6000/llama/ornith-1.5-122b-q4.ipynb), [`sage-a1.5-q4`](g4-rtx-pro6000/llama/sage-a1.5-q4.ipynb), [`qwen3.8-27b-bf16`](g4-rtx-pro6000/llama/qwen3.8-27b-bf16.ipynb), [`glm-5.3-flash-iq1s`](g4-rtx-pro6000/llama/glm-5.3-flash-iq1s.ipynb) |
| **vLLM** (`vllm/`) | [`qwen3.8-27b-fp8`](g4-rtx-pro6000/vllm/qwen3.8-27b-fp8.ipynb), [`sage-a1.5-fp8`](g4-rtx-pro6000/vllm/sage-a1.5-fp8.ipynb), [`nemotron-3.5-bolt-74b-w4a16`](g4-rtx-pro6000/vllm/nemotron-3.5-bolt-74b-w4a16.ipynb) |
| **SGLang** (`sglang/`) | [`qwen3.8-27b-w4a16-sglang`](g4-rtx-pro6000/sglang/qwen3.8-27b-w4a16-sglang.ipynb) |
| **Diffusers** (`diffusers/`) | [`qwen-image-2.1`](g4-rtx-pro6000/diffusers/qwen-image-2.1.ipynb), [`flux-schnell`](g4-rtx-pro6000/diffusers/flux-schnell.ipynb) |
| **Strata** (`strata/`) | [`flash-next-strata-iq3-s`](g4-rtx-pro6000/strata/flash-next-strata-iq3-s.ipynb), [`qwen36-strata-q6`](g4-rtx-pro6000/strata/qwen36-strata-q6.ipynb) |
| **Maya** (`maya/`) | [`glm-maya-s`](g4-rtx-pro6000/maya/glm-maya-s.ipynb) |
| **TensorFold** (`tensorfold/`) | [`qwen3.8-27b-exl3-2bpw`](g4-rtx-pro6000/tensorfold/qwen3.8-27b-exl3-2bpw.ipynb) |
| **ComfyUI** (`comfyui/`) | [`qwen-image-2.1-turbo-bf16-comfy`](g4-rtx-pro6000/comfyui/qwen-image-2.1-turbo-bf16-comfy.ipynb), [`qwen-image-2.1-bf16-comfy`](g4-rtx-pro6000/comfyui/qwen-image-2.1-bf16-comfy.ipynb), [`flux3-klein-comfy`](g4-rtx-pro6000/comfyui/flux3-klein-comfy.ipynb), [`wan22-5b-comfy`](g4-rtx-pro6000/comfyui/wan22-5b-comfy.ipynb), [`minimax-h3-g4-comfy`](g4-rtx-pro6000/comfyui/minimax-h3-g4-comfy.ipynb) |
| **Sparsa** (`sparsa/`) | [`minimax-m3.1-flash-iq2-kt`](g4-rtx-pro6000/sparsa/minimax-m3.1-flash-iq2-kt.ipynb) |
| **HyperQwen** (`hyperqwen/`) | [`hyperqwen-qwen3.8-27b-dflash2`](g4-rtx-pro6000/hyperqwen/hyperqwen-qwen3.8-27b-dflash2.ipynb) |
| **NInfer** (`ninfer/`) | [`ninfer-qwen3.8-flash-next`](g4-rtx-pro6000/ninfer/ninfer-qwen3.8-flash-next.ipynb), [`ninfer-qwen3.8-27b-nvfp4`](g4-rtx-pro6000/ninfer/ninfer-qwen3.8-27b-nvfp4.ipynb) |
| **Basalt** (`basalt/`) | [`basalt-flash-next-nvfp4-dq6k`](g4-rtx-pro6000/basalt/basalt-flash-next-nvfp4-dq6k.ipynb) |
| **Strata NVFP4** (`strata_nvfp4/`) | [`strata-nvfp4-flash-next-gptq`](g4-rtx-pro6000/strata_nvfp4/strata-nvfp4-flash-next-gptq.ipynb) |

## Other

- [`cpu/prepare_cpu.ipynb`](cpu/prepare_cpu.ipynb): on a cheap CPU runtime, download a model to Drive and build
  its engine for the GPU you will use, so the GPU notebook only restores them.
- [`other/unsloth/`](other/unsloth/): official Unsloth Studio.

Regenerate every notebook from the repo root (the hub code is packed into each one):

```bash
python3 -m notebook_code.make_engine_notebooks
```
