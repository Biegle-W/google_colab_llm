# Colab engine matrix for the four target models

Research synthesis for Google AI Pro / Colab Pro single-GPU runtimes (L4 ≈ 24 GB,
A100 ≈ 40/80 GB, G4 ≈ 96 GB). Sources: this user's
[`linux-llm-nvidia-public`](https://github.com/Biegle-W/linux-llm-nvidia-public) and
[`linux-llm-amd-public`](https://github.com/Biegle-W/linux-llm-amd-public) maintained
profiles, plus community/vendor reports (Winder.ai, ComputingForGeeks, Unsloth, hi5.sh
Colab G4 recipe, SGLang/vLLM cookbooks, Diffusers Colab NF4 writeups).

**Not portable to Colab (document only):** Strata and Maya are selected on the dual-GPU
AMD/NVIDIA workstations for Flash-Next / DeepSeek / GLM with SSD expert streaming. They
are not a Colab install path. HyperQwen is a patched vLLM stack that *can* run on a 24 GB
Ampere+ card (community Docker), but on Colab we use the same **W4A16 AutoRound**
checkpoint through stock vLLM / SGLang instead of shipping the HyperQwen container.

## Recommended engine by GPU

| Model | L4 (24 GB) | A100 40 GB (+ ~80 GB host) | A100 80 GB / G4 (~96 GB) |
|---|---|---|---|
| **Qwen3.8-27B** | **vLLM W4A16** (or llama.cpp UD-Q4) | **vLLM FP8** or **SGLang W4A16** | **vLLM FP8/NVFP4** or **SGLang**; HyperQwen-class INT4 for max context on 24–32 GB cards |
| **Qwen3.8-Flash-Next** | Not practical | **llama.cpp UD-IQ1_S** with n-gram table on CPU/SSD (needs high host RAM) | **llama.cpp UD-Q3_K_XL** + `-ot per_layer_token_embd.weight=CPU` (Colab G4 community ~63 GB VRAM @ 131K). Official vLLM/SGLang want multi-GPU (e.g. TP4). Workstation: **Strata** IQ3_XXS |
| **GLM-5.3-Flash** | Not practical (~100 GB+ combined) | Experimental **llama.cpp UD-IQ1_S** (~93 GB) if host RAM is high | **llama.cpp UD-IQ1_S** (or IQ3_XXS ~120 GB). Workstation: **Maya-S24** (selected). Official vLLM/SGLang want multi-GPU FP8 |
| **DeepSeek-V4.1-Flash-1017** | Not practical (~233 GB+) | Usually too large | Experimental **llama.cpp UD-IQ1_S** + Engram mmap/lazy (Unsloth). Workstation: **Maya** / **Strata** + UD-IQ1_S (selected) |
| **MiniMax-H3** | Experimental only | Experimental **ComfyUI** / **SGLang diffusion** layerwise offload (weights ~108 GB; host RAM bound) | Same; full-resident BF16 wants multi-GPU (SGLang Ulysses). **Not an LLM** — joint video+audio |
| **Qwen-Image-2.1** | **Diffusers NF4** (transformer+text encoder) or Turbo | **Diffusers BF16** / ComfyUI Turbo (your workflows) | Diffusers BF16, **SGLang** image, or **vLLM-Omni** |

## Evidence notes (short)

### Qwen3.8-27B
- Your NVIDIA registry selects **HyperQwen-vLLM** (AutoRound INT4, ~86K ctx) and **SGLang**
  (~57K) as maintained dense profiles; TensorFold is also selected for short-context.
- Your AMD dual-XTX table: SGLang TP2 DFlash2 leads code decode; vLLM TP2 MTP3 is close.
- Public H100 concurrency bake-off (Winder.ai): **SGLang** and **vLLM** scale; llama.cpp /
  Ollama do not for multi-user. For **one user on 24 GB**, llama.cpp Q4 or W4A16 vLLM is the
  practical path (ComputingForGeeks / HyperQwen community).

### Qwen3.8-Flash-Next
- 125B-A6B + ~51B n-gram table. Official SGLang/vLLM recipes are multi-GPU.
- Unsloth / hi5 Colab **G4** recipe: llama.cpp `UD-Q3_K_XL`, n-gram on CPU, ~90 GB download,
  ~63 GB VRAM.
- Your workstation: **Strata** IQ3_XXS (selected) and **llama.cpp IQ1_S** (enabled API entry).

### MiniMax-H3
- Video/audio DiT (~108 GB), not a chat model. Your AMD/NVIDIA trees keep **ComfyUI**
  workflows (`minimax-h3-*.json`). Vendor path: **SGLang** diffusion; community also
  **vLLM-Omni**. Single Colab GPU implies layerwise/CPU offload and lots of host RAM —
  treat as experimental.

### GLM-5.3-Flash
- Your NVIDIA registry: **Maya-S24** selected (~95 GB Maya quant, 64K context on the workstation).
- Colab path: Unsloth `GLM-5.3-Flash-GGUF` **UD-IQ1_S** (~93 GB) via llama.cpp with `--fit`; needs ~100 GB combined RAM/VRAM and a build that understands GLM-5.3 / ox-alpha (Unsloth llama.cpp PR).
- Official **vLLM** / **SGLang** FP8 recipes are multi-GPU (e.g. TP8 H200).

### DeepSeek-V4.1-Flash-1017
- Your NVIDIA registry: **Maya** and **Strata** + Unsloth **UD-IQ1_S** (~233 GB) selected.
- Colab path: same Unsloth GGUF with Engram on Drive + mmap/lazy; expect a very large download and slow first load. Prefer the workstation unless you have G4 + lots of host RAM/disk.
- Needs Unsloth/deepseek41 llama.cpp (not the older Qwen-only pin alone).

### Qwen-Image-2.1
- Official: Diffusers `QwenImage21Pipeline` (BF16). Your repos: ComfyUI Turbo workflows.
- Colab T4 community: NF4 on transformer+text encoder ≈ 10.7 GiB, 512² in ~22 s.
- Higher GPUs: BF16 / Turbo; SGLang and vLLM-Omni have Day-0 image recipes.

## Colab hub mapping

| Catalog id | Engine | Role |
|---|---|---|
| `qwen3.8-27b-q4` | llama.cpp | L4-friendly GGUF chat |
| `qwen3.8-27b-w4a16` | vLLM | L4+/A100 W4A16 (HyperQwen-compatible weights) |
| `qwen3.8-27b-w4a16-sglang` | SGLang | Same weights, SGLang when installed |
| `qwen3.8-27b-fp8` | vLLM | A100/G4 FP8 |
| `flash-next-iq1s` | llama.cpp | Flash-Next on A100-class + high RAM |
| `flash-next-q3` | llama.cpp | Flash-Next on G4 / 80 GB+ |
| `glm-5.3-flash-iq1s` | llama.cpp | GLM-5.3-Flash (~93 GB); Maya on workstation |
| `glm-5.3-flash-iq3` | llama.cpp | Higher-quality GLM quant (~120 GB) |
| `deepseek-v4.1-flash-1017-iq1s` | llama.cpp | DeepSeek-V4.1-Flash-1017 (~233 GB); Maya/Strata on workstation |
| `qwen-image-2.1` | Diffusers | Image BF16 |
| `qwen-image-2.1-nf4` | Diffusers | Image NF4 for ≤24 GB |
| `minimax-h3` | (experimental) | Video; see notebook Comfy/SGLang notes |

`recommend.for_gpu(vram_gb)` returns the preferred prefetch list and engines for a runtime.
