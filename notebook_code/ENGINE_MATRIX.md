# Colab engine matrix for the target models

Research synthesis for Google AI Pro / Colab Pro single-GPU runtimes (L4 ≈ 24 GB,
A100 ≈ 40/80 GB, G4 ≈ 96 GB). Sources: **official** Strata / Project Maya / Sparsa /
TensorFold / Unsloth / vLLM / SGLang / ComfyUI docs, plus Colab forum reports
(Qiita Strata+L4, note.com Strata+A100) and the workstation
`linux-llm-nvidia-public` maintained profiles.

**Colab layout:** one notebook per runtime, engine and model under
`notebooks/<runtime>/<engine>/<model>.ipynb` (index: `notebooks/README.md`). Each installs only that
engine and serves only that model. Settings: `settings/settings_<engine>.json`.

## Recommended engine by model (researched)

| Model | Best on Colab (researched) | Also OK | Avoid on single Colab GPU |
|---|---|---|---|
| **Qwen3.8-27B** | **vLLM W4A16/FP8** or **SGLang W4A16** or **TensorFold EXL3+DFlash2**; on G4 also **HyperQwen** / **NInfer** | llama.cpp Q4 | HyperQwen speculation on A100 (sm80 fault) |
| **Qwen3.8-Flash-Next** | **Strata** (ISTA / Gyro / Swift) | llama.cpp Unsloth GGUF | Official vLLM/SGLang (multi-GPU) |
| **Qwen3.6-35B-A3B** | **Strata** (`--family qwen36`) or vLLM FP8 | llama.cpp Unsloth | — |
| **GLM-5.3-Flash** | **Maya** (Maya-S24 / Maya-S / Maya-XS) | llama.cpp Unsloth GGUF (experimental) | Official vLLM/SGLang (TP4–8) |
| **DeepSeek-V4.1-Flash-1017** | **Maya** (`maya-ds4`) or **Strata** (`strata-dsv41`) | Unsloth llama.cpp GGUF | Multi-GPU FP8 stacks |
| **MiniMax-M3.1-Flash** | **Sparsa** (IQ2_XS-D3 / IQ2_KT) | — | stock llama.cpp / Diffusers |
| **Nemotron-3.5-Bolt-74B** | **llama.cpp native IQ2** (workstation selected) | vLLM W4A16 experimental | — |
| **Sage-A1.5 / Ornith-1.5** | llama.cpp Q4 / vLLM FP8 (Sage) | — | Ornith needs G4-class VRAM |
| **Qwen-Image-2.1** | **ComfyUI**: Turbo INT8 + INT8 encoder (L4); Turbo / base BF16 (A100, G4) | Diffusers NF4 / BF16 | — |
| **FLUX.3 Klein / Wan 2.2** | **ComfyUI** | — | Diffusers hub (not catalogued) |
| **MiniMax-H3** | **ComfyUI** (`/v1/videos/generations`): pruned INT8 (L4), INT8 + INT8 encoder (A100), INT8 + NVFP4 encoder (G4); Turbo 8-step LoRA | — | Diffusers (no H3 pipeline) |

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

### HyperQwen ([syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen))
- Qwen3.8-27B only (`dbirks/Qwen3.8-27B-W4A16-AutoRound`), a patch series on pinned vLLM 0.32.0.
- Install: pinned checkout + venv (`vllm==0.32.0`, FlashInfer cubin, CUDA 13 nvcc wheels) + `patches/apply.sh`;
  the whole tree is cached on Drive as one tarball.
- Prepare: `prepare/quant_lm_head.py`, `quant_embed.py`, `quant_mtp.py`, `build_draft_vocab.py` rewrite the
  checkpoint in place, so the hub runs them on a local copy (`<local models>/.hyperqwen/<id>`), once per runtime.
- Serve: `single-user/start_qwen.sh` (SPEC=mtp|dflash2) or `batch/start_qwen.sh`, with `MODEL`, `PORT`,
  `MAX_LEN`; the server name is fixed at `qwen3.8-27b`, so the hub rewrites the request's `model` field.
- GPUs: tuned on sm86 (3090), reproduced on sm120. sm80 (A100) has an open speculative-decoding fault
  (#98, #72), so only the batch entry is offered there. Needs a 24 GB card: the L4 (22.5 GiB) is just under.

### NInfer ([Neroued/ninfer](https://github.com/Neroued/ninfer))
- Native C++/CUDA engine for selected models: Qwen3.8-27B (NVFP4 / group-wise int), Qwen3.6-35B-A3B,
  Qwen3.8-Flash-Next (CPU expert offload). GLM-5.3-Flash (178 GiB) is not catalogued.
- Build: `cmake --preset rtx5090-release` targets **sm_120a only**, so Colab G4 only. Needs CUDA 13
  (driver 580+), hwloc ≥ 2.5, liburing ≥ 2.5, libcurl ≥ 7.85; the installer builds liburing / libcurl
  from source when Ubuntu's are older, and bundles them with the CUDA 13 runtime in the cached `lib/`.
- Serve: `ninfer-serve ARTIFACT.ninfer --host 127.0.0.1 --port P --model-id ID --max-context N --kv-dtype fp8
  --spec mtp --draft-tokens 3 --proposal-head optimized`.

### Community research (2026-11-08): RTX PRO 6000 (= Colab G4) and A100

From lcz.me (抡锤者, LLM board), Level1Techs, GitHub and Hugging Face model cards. Numbers are as reported
by their authors, not re-measured here.

| Model / engine | Setup | Reported | Source |
|---|---|---|---|
| Qwen3.5-122B-A10B Q4_K_M + MTP, llama.cpp | 1× RTX PRO 6000, fully on GPU | 146.5 tok/s (96.9 without MTP), to 262K ctx | localllm.in, May 2026 |
| gpt-oss-120b MXFP4, llama.cpp | 1× RTX PRO 6000 | 196–208 tok/s | llama.cpp discussion #15396; localllm.in |
| GLM-5.3-Flash AJ-IQ2_XXS, llama.cpp | 1× RTX PRO 6000, fully resident | 69.9 tok/s (31.7 at 32K depth) | aj9o9/GLM-5.3-Flash-GGUF card |
| Qwen3.8-Flash-Next IQ3_S, Strata vs llama.cpp | 1× RTX PRO 6000, 4K prompt | ~205 vs ~103 tok/s (others report ~100) | llamaperf.com aggregation |
| Qwen3.8-Flash-Next, **Basalt** (Strata fork for Blackwell sm_120) | 5070 Ti, same model as Strata | 1.26–1.42× Strata decode, 1.39× prefill | lcz.me topic 2892 (2026-11-07) |
| GLM-5.3-Flash Maya-L IQ3_S (156 GB), Maya | 2× RTX PRO 6000 | ~127–137 tok/s at 51K ctx | lcz.me topic 2882 |
| DeepSeek-V4.1-Flash-1017 EXL3 2.7 bpw, Strata-DS fork | 2× RTX PRO 6000, Engram on NVMe | 80.8 tok/s with DSpark, 524K ctx | lcz.me topic 2452 |

Not added to the catalog (not requested): the Qwen3.5-122B, gpt-oss-120b and GLM AJ-IQ2_XXS rows above.
Candidates not yet integrated: **Basalt** (github.com/jesdga95/basalt; sm_120 + CUDA 13, which the G4 has;
packs at huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt, default NVFP4-DQ6_K 131.7 GB, and a DeepSeek
1017 repo) and NVIDIA **Flare** (github.com/NVIDIA/spark-local-inference; research prototype, no weights or
install steps published, so not pursued).

### Measured on Colab (G4, 2026-11-07, via the Colab CLI)

G4 runtime: RTX PRO 6000 Blackwell 96 GB, driver 580.82 (CUDA 13.0), nvcc 13.0, 48 threads, 176 GB RAM,
253 GB disk, Python 3.13, Ubuntu 24.04. CPU runtime: 2 threads, 12 GB RAM, 242 GB disk, no CUDA toolkit.

| Step | HyperQwen (`-dflash2`) | NInfer (`-nvfp4`) |
|---|---|---|
| Build | 14 min on CPU runtime, ~10 min on G4; 15 GB tarball | ~11 min on G4 (48 threads); 326 MB tarball |
| Model download (straight to Drive dir) | 20.8 GB | 23.7 GB |
| First request on a fresh runtime | 304 s (prepare scripts + torch.compile + graphs) | 67 s |
| Same runtime, warm caches | 39 s | — |
| Decode, 1 request, temperature 0, code prompt | ~315 tok/s | ~187 tok/s |

Both engines think by default; the hub starts them with thinking off (as the reference stack does), and a
request can turn it on with `chat_template_kwargs: {"enable_thinking": true}`.

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
| `qwen-image-2.1-turbo-comfy` (L4), `qwen-image-2.1-turbo-bf16-comfy`, `qwen-image-2.1-bf16-comfy` (A100 / G4), `flux3-klein-comfy` | **ComfyUI** | API images |
| `minimax-h3-l4-comfy`, `minimax-h3-comfy` (A100), `minimax-h3-g4-comfy` | **ComfyUI** | API video + audio |
| `wan22-5b-comfy` | **ComfyUI** | Video prefetch (UI workflow) |
| `minimax-m3.1-flash-iq2-*` | **Sparsa** | MiniMax-M3.1-Flash |
| `hyperqwen-qwen3.8-27b-{mtp,dflash2,batch}` | **HyperQwen** | Qwen3.8-27B single-user / batch |
| `ninfer-qwen3.8-27b{,-nvfp4}`, `ninfer-qwen3.6-35b-a3b`, `ninfer-qwen3.8-flash-next` | **NInfer** | G4 only |
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
