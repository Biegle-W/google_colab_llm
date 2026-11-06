# Colab engine matrix for the target models

Research synthesis for Google AI Pro / Colab Pro single-GPU runtimes (L4 ≈ 24 GB,
A100 ≈ 40/80 GB, G4 ≈ 96 GB). Sources: **official** Strata / Project Maya / Sparsa /
TensorFold / Unsloth / vLLM / SGLang / ComfyUI docs, plus Colab forum reports
(Qiita Strata+L4, note.com Strata+A100) and the workstation
`linux-llm-nvidia-public` maintained profiles.

**Colab layout:** one notebook per engine under `notebooks/<engine>/` (index:
`notebooks/llm_hub_colab.ipynb`). Each installs only that engine and lists only its catalog
models. Settings: `config/settings_<engine>.json`.

## Recommended engine by model (researched)

| Model | Best on Colab (researched) | Also OK | Avoid on single Colab GPU |
|---|---|---|---|
| **Qwen3.8-27B** | **vLLM W4A16/FP8** or **SGLang W4A16** or **TensorFold EXL3+DFlash2** | llama.cpp Q4 | — |
| **Qwen3.8-Flash-Next** | **Strata** (ISTA / Gyro / Swift) | llama.cpp Unsloth GGUF | Official vLLM/SGLang (multi-GPU) |
| **Qwen3.6-35B-A3B** | **Strata** (`--family qwen36`) or vLLM FP8 | llama.cpp Unsloth | — |
| **GLM-5.3-Flash** | **Maya** (Maya-S24 / Maya-S / Maya-XS) | llama.cpp Unsloth GGUF (experimental) | Official vLLM/SGLang (TP4–8) |
| **DeepSeek-V4.1-Flash-1017** | **Maya** (`maya-ds4`) or **Strata** (`strata-dsv41`) | Unsloth llama.cpp GGUF | Multi-GPU FP8 stacks |
| **MiniMax-M3.1-Flash** | **Sparsa** (IQ2_XS-D3 / IQ2_KT) | — | stock llama.cpp / Diffusers |
| **Nemotron-3.5-Bolt-74B** | **llama.cpp native IQ2** (workstation selected) | vLLM W4A16 experimental | — |
| **Sage-A1.5 / Ornith-1.5** | llama.cpp Q4 / vLLM FP8 (Sage) | — | Ornith needs G4-class VRAM |
| **Qwen-Image-2.1** | **ComfyUI Turbo** or **Diffusers** NF4 / BF16 | — | — |
| **FLUX.3 Klein / Wan 2.2** | **ComfyUI** | — | Diffusers hub (not catalogued) |
| **MiniMax-H3** | **ComfyUI** prefetch + UI | — | Diffusers hub server |

## Official references

### TensorFold ([ashhart/TensorFold](https://github.com/ashhart/TensorFold))
- Workstation profile: `qwen27b-tensorfold` — Mia EXL3 2.0bpw + DFlash2 4-bit @ 12K.
- Colab install: official `tensorfold-*-linux-x86_64.tar.gz` release (`bin/tensorfold-native`).
- Serve: `tensorfold-native serve MODEL --backend cuda --drafter DRAFTER --drafter-bits 4 --name …`.
- OpenAI API: `http://127.0.0.1:<port>/v1`.

### ComfyUI ([Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI))
- Workstation workflows: Qwen-Image-2.1 Turbo, FLUX.3 Klein, Wan 2.2, LTX-2.3 (UI JSON under `workflows/`).
- Colab path: stock ComfyUI + OpenAI wrapper (`comfyui_server`) using `*.api.json` loader graphs.
- Custom `NvidiaResident*` nodes stay optional (workstation dual-GPU); Colab uses stock loaders.

### Sparsa ([LyJonathan/Sparsa](https://github.com/LyJonathan/Sparsa))
- MiniMax-M3.1-Flash only. Not stock llama.cpp.
- Install: clone + linux CUDA `strata-minimax` release tarball into `bin/`.
- Serve: `python sparsa.py serve --model DIR --no-api-key --host 127.0.0.1`.

### Strata ([Niko1221/Strata](https://github.com/Niko1221/Strata))
- Primary: **Qwen3.8-Flash-Next** (ISTA; also Gyro / Swift / Unsloth).
- Also: **Qwen3.6-35B-A3B** (`--family qwen36`), DeepSeek (`--family deepseek41`).
- Headless: `strata launch … --yes --no-tui`.

### Project Maya ([mw00/project-maya](https://github.com/mw00/project-maya))
- Primary: **GLM-5.3-Flash** + **DeepSeek-V4.1-Flash-1017**.
- Setup: `./setup.sh --model glm-maya-s24 --yes`.

### Unsloth / llama.cpp
- Bolt native IQ2, Sage, Ornith, and MoE GGUF fallbacks when Strata/Maya/Sparsa are unavailable.

### vLLM / SGLang
- Dense Qwen W4A16/FP8 on single Colab GPU.
- Official GLM / Flash-Next / DeepSeek recipes remain multi-GPU — not the Colab path.

## Colab hub mapping

| Catalog id | Engine | Role |
|---|---|---|
| `qwen3.8-27b-exl3-2bpw` | **TensorFold** | Mia EXL3 + DFlash2 |
| `qwen-image-2.1-turbo-comfy`, `flux3-klein-comfy` | **ComfyUI** | Interactive / API images |
| `wan22-5b-comfy`, `minimax-h3-comfy` | **ComfyUI** | Video prefetch |
| `minimax-m3.1-flash-iq2-*` | **Sparsa** | MiniMax-M3.1-Flash |
| `flash-next-strata-*` / `gyro-*` / `swift-*` | **Strata** | Flash-Next variants |
| `qwen36-strata-*` | **Strata** | Qwen3.6-35B-A3B |
| `deepseek-strata-ist-2.00` | **Strata** | DeepSeek ISTA 2.00bpw |
| `glm-maya-*` / `deepseek-maya-*` | **Maya** | GLM / DeepSeek |
| `nemotron-3.5-bolt-74b-iq2` | llama.cpp | Bolt (selected) |
| `sage-a1.5-*`, `ornith-1.5-*` | llama / vLLM | Promoted dense MoEs |
| `qwen3.8-27b-*` | vLLM / SGLang / llama | Dense 27B |
| `qwen-image-2.1*` | Diffusers | Images (API) |

`recommend.for_engine(engine, vram_gb)` returns the recommended models for one notebook, best first; the
notebook's `MODEL = "recommended"` uses the first one that fits, and only that one model is downloaded.
