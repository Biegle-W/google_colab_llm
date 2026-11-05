# Colab engine matrix for the target models

Research synthesis for Google AI Pro / Colab Pro single-GPU runtimes (L4 ≈ 24 GB,
A100 ≈ 40/80 GB, G4 ≈ 96 GB). Sources: **official** Strata / Project Maya / Unsloth /
vLLM / SGLang docs, plus Colab forum reports (Qiita Strata+L4, note.com Strata+A100).

**Colab layout:** one notebook per engine — `llama_hub_colab.ipynb`, `vllm_hub_colab.ipynb`,
`sglang_hub_colab.ipynb`, `diffusers_hub_colab.ipynb`, **`strata_hub_colab.ipynb`**,
**`maya_hub_colab.ipynb`**. Each installs only that engine and lists only its catalog models.
Settings: `config/settings_<engine>.json`.

## Recommended engine by model (researched)

| Model | Best on Colab (researched) | Also OK | Avoid on single Colab GPU |
|---|---|---|---|
| **Qwen3.8-27B** | **vLLM W4A16/FP8** or **SGLang W4A16** | llama.cpp Q4 | — |
| **Qwen3.8-Flash-Next** | **Strata** (ISTA Q2_0 / IQ3_XXS / IQ3_S) | llama.cpp Unsloth GGUF | Official vLLM/SGLang (multi-GPU) |
| **GLM-5.3-Flash** | **Maya** (Maya-S24 / Maya-S / Maya-XS) | llama.cpp Unsloth GGUF (experimental) | Official vLLM/SGLang (TP4–8) |
| **DeepSeek-V4.1-Flash-1017** | **Maya** (`maya-ds4`) or **Strata** (`strata-dsv41`) | Unsloth llama.cpp GGUF | Multi-GPU FP8 stacks |
| **Qwen-Image-2.1** | **Diffusers** NF4 / BF16 | — | — |
| **MiniMax-H3** | Prefetch → ComfyUI / SGLang diffusion | — | Diffusers hub server |

## Official references

### Strata ([Niko1221/Strata](https://github.com/Niko1221/Strata))
- Primary model: **Qwen3.8-Flash-Next** (ISTA-DASLab packs; Unsloth / Gyro alternatives).
- DeepSeek: `strata launch --family deepseek41 --model ist-das-2.00bpw` (`docs/DEEPSEEK41.md`).
- Headless Colab-friendly: `strata launch … --yes --no-tui` (`docs/TERMINAL_LAUNCH.md`, `docs/AI_SETUP.md`).
- OpenAI API: `http://127.0.0.1:<port>/v1`.
- **Colab forums:** Qiita L4 + Strata Flash-Next; note.com A100 + Strata + local Pi client.

### Project Maya ([mw00/project-maya](https://github.com/mw00/project-maya))
- Primary models: **GLM-5.3-Flash** (`glm-maya-s24` / `s` / `xs` / …) and **DeepSeek-V4.1-Flash-1017** (`deepseek-maya-xs` / `s`).
- Derived from Strata; OpenAI-compatible dashboard/API on port 8080 by default.
- Quants: [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF),
  [peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF).
- Setup: `./setup.sh --model glm-maya-s24 --yes` (`docs/SETUP.md`).

### Unsloth / llama.cpp
- Documented fallback for DeepSeek / Flash-Next GGUFs when Strata/Maya are unavailable.
- Not the preferred path when Strata/Maya install succeeds.

### vLLM / SGLang (GLM)
- [vLLM GLM-5.3-Flash recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) and
  [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash) target
  **multi-GPU** FP8 (TP4–8). Not used as the Colab single-GPU path for GLM/DeepSeek/Flash-Next.

## Colab hub mapping

| Catalog id | Engine | Role |
|---|---|---|
| `flash-next-strata-q2` / `iq3-xxs` / `iq3-s` | **Strata** | Flash-Next (official ISTA packs) |
| `deepseek-strata-ist-2.00` | **Strata** | DeepSeek ISTA 2.00bpw |
| `glm-maya-s24` / `s` / `xs` | **Maya** | GLM-5.3-Flash official quants |
| `deepseek-maya-xs` / `s` | **Maya** | DeepSeek official Maya quants |
| `qwen3.8-27b-*` | vLLM / SGLang / llama | Dense 27B |
| `flash-next-iq1s` / `q3` | llama.cpp | Flash-Next fallback |
| `glm-5.3-flash-iq1s` / `iq3` | llama.cpp | GLM fallback |
| `deepseek-v4.1-flash-1017-iq1s` | llama.cpp | DeepSeek fallback |
| `qwen-image-2.1*` | Diffusers | Images |

`recommend.for_engine(engine, vram_gb)` returns the preferred prefetch list for one notebook.
