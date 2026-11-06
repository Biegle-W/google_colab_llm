"""Pick catalog ids for one Colab engine (or a multi-engine overview).

Each hub notebook locks to a single engine. Recommendations follow ENGINE_MATRIX.md
plus official Strata / Project Maya / TensorFold / Sparsa / ComfyUI docs and Colab forum recipes.
"""
from . import catalog

ENGINES = ("llama", "vllm", "sglang", "diffusers", "strata", "maya", "tensorfold", "comfyui", "sparsa")

# Preferred catalog ids per engine × GPU band (best-first).
_ENGINE_PREFETCH = {
    "llama": {
        "t4": [],
        "l4": ["qwen3.8-27b-q4", "nemotron-3.5-bolt-74b-iq2"],
        "a100_40": ["qwen3.8-27b-q8", "nemotron-3.5-bolt-74b-iq2", "sage-a1.5-q4", "flash-next-iq1s"],
        "a100_80": ["qwen3.8-27b-q8", "nemotron-3.5-bolt-74b-iq2", "sage-a1.5-q4", "flash-next-q3"],
        "g4": ["nemotron-3.5-bolt-74b-iq2", "flash-next-q3", "ornith-1.5-122b-q4", "sage-a1.5-q4",
               "qwen3.8-27b-bf16", "glm-5.3-flash-iq1s"],
    },
    "vllm": {
        "t4": [],
        "l4": ["qwen3.8-27b-w4a16", "qwen3-14b-fp8"],
        "a100_40": ["qwen3.8-27b-fp8", "qwen3.8-27b-w4a16", "qwen3.6-35b-a3b-fp8"],
        "a100_80": ["qwen3.8-27b-fp8", "sage-a1.5-fp8", "nemotron-3.5-bolt-74b-w4a16",
                    "qwen3.8-27b-bf16-vllm"],
        "g4": ["qwen3.8-27b-fp8", "sage-a1.5-fp8", "nemotron-3.5-bolt-74b-w4a16",
               "qwen3.8-27b-bf16-vllm"],
    },
    "sglang": {
        "t4": [],
        "l4": ["qwen3.8-27b-w4a16-sglang"],
        "a100_40": ["qwen3.8-27b-w4a16-sglang"],
        "a100_80": ["qwen3.8-27b-w4a16-sglang"],
        "g4": ["qwen3.8-27b-w4a16-sglang"],
    },
    "diffusers": {
        "t4": ["qwen-image-2.1-nf4", "sd15"],
        "l4": ["qwen-image-2.1-nf4", "sdxl-turbo", "sdxl-base"],
        "a100_40": ["qwen-image-2.1", "flux-schnell", "sdxl-turbo"],
        "a100_80": ["qwen-image-2.1", "flux-schnell"],
        "g4": ["qwen-image-2.1", "flux-schnell", "minimax-h3"],
    },
    "strata": {
        "t4": [],
        "l4": ["flash-next-gyro-s", "flash-next-strata-q2", "flash-next-swift-iq2-xs", "qwen36-strata-q4"],
        "a100_40": ["flash-next-strata-iq3-xxs", "flash-next-gyro-m", "flash-next-swift-iq3-xxs",
                    "qwen36-strata-q4", "flash-next-strata-q2"],
        "a100_80": ["flash-next-strata-iq3-s", "flash-next-strata-iq3-xxs", "flash-next-gyro-m",
                    "qwen36-strata-q6"],
        "g4": ["flash-next-strata-iq3-s", "flash-next-strata-iq3-xxs", "qwen36-strata-q6",
               "deepseek-strata-ist-2.00"],
    },
    "maya": {
        "t4": [],
        "l4": ["glm-maya-xs", "glm-maya-s24"],
        "a100_40": ["glm-maya-s24", "glm-maya-s", "glm-maya-xs"],
        "a100_80": ["glm-maya-s", "glm-maya-s24", "deepseek-maya-xs"],
        "g4": ["glm-maya-s", "glm-maya-s24", "deepseek-maya-xs", "deepseek-maya-s"],
    },
    "tensorfold": {
        "t4": [],
        "l4": ["qwen3.8-27b-exl3-2bpw"],
        "a100_40": ["qwen3.8-27b-exl3-2bpw"],
        "a100_80": ["qwen3.8-27b-exl3-2bpw"],
        "g4": ["qwen3.8-27b-exl3-2bpw"],
    },
    "comfyui": {
        "t4": ["qwen-image-2.1-turbo-comfy"],
        "l4": ["qwen-image-2.1-turbo-comfy", "flux3-klein-comfy"],
        "a100_40": ["qwen-image-2.1-turbo-comfy", "flux3-klein-comfy", "wan22-5b-comfy"],
        "a100_80": ["qwen-image-2.1-turbo-comfy", "flux3-klein-comfy", "wan22-5b-comfy"],
        "g4": ["qwen-image-2.1-turbo-comfy", "flux3-klein-comfy", "wan22-5b-comfy", "minimax-h3-comfy"],
    },
    "sparsa": {
        "t4": [],
        "l4": [],
        "a100_40": [],
        "a100_80": ["minimax-m3.1-flash-iq2-kt"],
        "g4": ["minimax-m3.1-flash-iq2-xs-d3", "minimax-m3.1-flash-iq2-kt"],
    },
}

_ENGINE_NOTES = {
    "llama": {
        "t4": [
            "T4: llama.cpp chat targets need L4+ (AI Pro / Colab Pro).",
            "Use Diffusers/ComfyUI for images; Strata/Maya/Sparsa for huge MoEs.",
        ],
        "l4": [
            "L4: UD-Q4 Qwen3.8-27B and native IQ2 Nemotron Bolt (fit/offload).",
            "For Flash-Next prefer Strata; for GLM prefer Maya.",
        ],
        "a100_40": [
            "A100 40 GB: Qwen Q8, Bolt IQ2, Sage-A1.5 Q4; Flash-Next IQ1_S is a fallback vs Strata.",
        ],
        "a100_80": [
            "A100 80 GB: Bolt + Sage + Flash-Next Q3; Ornith needs G4-class VRAM.",
        ],
        "g4": [
            "G4: Bolt IQ2, Ornith-1.5-122B Q4, Sage, large MoE GGUFs. Prefer Strata/Maya/Sparsa notebooks for streaming MoEs.",
        ],
    },
    "vllm": {
        "t4": ["T4: vLLM targets need L4+."],
        "l4": [
            "L4: W4A16 AutoRound Qwen3.8-27B or Qwen3-14B FP8.",
            "Flash-Next / GLM / DeepSeek / MiniMax-M3.1 official vLLM recipes are multi-GPU — use Strata/Maya/Sparsa.",
        ],
        "a100_40": ["A100 40 GB: Qwen3.8-27B FP8 or W4A16; Qwen3.6-35B-A3B FP8."],
        "a100_80": ["A100 80 GB: FP8 / BF16 dense chat; Sage A1.5 FP8; experimental Bolt W4A16."],
        "g4": ["G4: Qwen3.8-27B FP8 or BF16; Sage A1.5 FP8; experimental Bolt W4A16."],
    },
    "sglang": {
        "t4": ["T4: SGLang targets need L4+."],
        "l4": [
            "L4+: Qwen3.8-27B W4A16 via SGLang.",
            "Flash-Next / GLM / DeepSeek multi-GPU SGLang recipes → use Strata/Maya on Colab.",
        ],
        "a100_40": ["A100: SGLang W4A16 for Qwen3.8-27B."],
        "a100_80": ["A100 80 GB / G4: same W4A16 chat entry."],
        "g4": ["G4: Qwen3.8-27B W4A16 via SGLang."],
    },
    "diffusers": {
        "t4": ["T4: Qwen-Image-2.1 NF4 or SD 1.5.", "MiniMax-H3 is video — prefer ComfyUI notebook."],
        "l4": ["L4: Qwen-Image-2.1 NF4, SDXL Turbo / base. Interactive editing → ComfyUI."],
        "a100_40": ["A100 40 GB: Qwen-Image-2.1 BF16, FLUX.1-schnell."],
        "a100_80": ["A100 80 GB: Qwen-Image-2.1 BF16, FLUX.1-schnell."],
        "g4": ["G4: Qwen-Image-2.1 BF16, FLUX; MiniMax-H3 prefetch — prefer ComfyUI for video."],
    },
    "strata": {
        "t4": [
            "T4: Strata Flash-Next needs L4+ and high host RAM (forum L4/A100 recipes).",
        ],
        "l4": [
            "L4: Gyro-S / ISTA Q2_0 / Swift IQ2_XS / Qwen3.6 Q4 (memory-tier options).",
            "Official: `strata launch --family … --model … --yes --no-tui`.",
        ],
        "a100_40": [
            "A100 40 GB: Strata IQ3_XXS (ISTA) or Gyro-M / Swift IQ3_XXS / Qwen3.6 Q4.",
        ],
        "a100_80": [
            "A100 80 GB: Strata IQ3_S (ISTA recommended) or IQ3_XXS; Qwen3.6 Q6.",
        ],
        "g4": [
            "G4: Strata IQ3_S / IQ3_XXS for Flash-Next; DeepSeek ISTA 2.00bpw (~257 GB) via strata-dsv41.",
        ],
    },
    "maya": {
        "t4": ["T4: Maya GLM/DeepSeek need L4+ and large host RAM/disk."],
        "l4": [
            "L4: Maya-XS / Maya-S24 for GLM-5.3-Flash (official ≤24 GB VRAM pick).",
            "Install: clone mw00/project-maya; `./setup.sh --model glm-maya-s24 --yes`.",
        ],
        "a100_40": [
            "A100 40 GB: Maya-S24 or Maya-S for GLM-5.3-Flash.",
        ],
        "a100_80": [
            "A100 80 GB: Maya-S for GLM; DeepSeek Maya-XS if disk allows (~281 GB).",
        ],
        "g4": [
            "G4: best Colab band for Maya GLM + DeepSeek Maya-XS/S (huge Drive downloads).",
        ],
    },
    "tensorfold": {
        "t4": ["T4: TensorFold EXL3 27B needs L4+ (~14 GB+)."],
        "l4": [
            "L4+: Qwen3.8-27B Mia EXL3 2.0bpw + DFlash2 4-bit (workstation qwen27b-tensorfold).",
            "Official: `tensorfold-native serve … --drafter … --drafter-bits 4` (TensorFold 1.2.2).",
        ],
        "a100_40": ["A100: same EXL3 + DFlash2 entry; raise context toward 16K if VRAM allows."],
        "a100_80": ["A100 80 GB / G4: EXL3 27B with headroom for longer context."],
        "g4": ["G4: EXL3 27B + DFlash2; still lower precision than AutoRound W4A16."],
    },
    "comfyui": {
        "t4": ["T4: Qwen-Image-2.1 Turbo via ComfyUI (stock API workflow)."],
        "l4": ["L4: Qwen-Image-2.1 Turbo + FLUX.3 Klein (workstation Comfy entries)."],
        "a100_40": ["A100: image entries + Wan 2.2 5B video prefetch."],
        "a100_80": ["A100 80 GB: image + Wan video prefetch; MiniMax-H3 needs G4 disk."],
        "g4": ["G4: full Comfy set including MiniMax-H3 prefetch; video OpenAI route is 501 (use UI)."],
    },
    "sparsa": {
        "t4": ["T4: MiniMax-M3.1-Flash via Sparsa needs high-RAM A100/G4-class hosts."],
        "l4": ["L4: Sparsa MiniMax packs need ~100 GB+ combined memory — use A100 80 / G4."],
        "a100_40": ["A100 40 GB: still tight for Sparsa MiniMax; prefer A100 80 / G4 + large disk."],
        "a100_80": [
            "A100 80 GB: IQ2_KT (~124 GB pack) with SSD expert streaming — experimental.",
            "Official: `python sparsa.py serve --model … --no-api-key`.",
        ],
        "g4": [
            "G4: IQ2_XS-D3 (default Sparsa pick, ~146 GB) or IQ2_KT. Huge Drive download.",
        ],
    },
}

# Hub notebooks live under notebooks/<engine>/…
ENGINE_META = {
    "llama": {
        "title": "llama.cpp",
        "notebook": "notebooks/llama/llama_hub_colab.ipynb",
        "kind": "chat",
        "summary": "GGUF: Qwen3.8-27B, Bolt IQ2, Sage, Ornith, Flash-Next / GLM / DeepSeek fallbacks",
    },
    "vllm": {
        "title": "vLLM",
        "notebook": "notebooks/vllm/vllm_hub_colab.ipynb",
        "kind": "chat",
        "summary": "Safetensors chat: Qwen3.8-27B W4A16/FP8/BF16, Sage, Bolt W4A16, …",
    },
    "sglang": {
        "title": "SGLang",
        "notebook": "notebooks/sglang/sglang_hub_colab.ipynb",
        "kind": "chat",
        "summary": "SGLang chat: Qwen3.8-27B W4A16 AutoRound",
    },
    "diffusers": {
        "title": "Diffusers",
        "notebook": "notebooks/diffusers/diffusers_hub_colab.ipynb",
        "kind": "image",
        "summary": "Text-to-image: Qwen-Image-2.1, FLUX, SDXL (+ MiniMax-H3 prefetch)",
    },
    "strata": {
        "title": "Strata",
        "notebook": "notebooks/strata/strata_hub_colab.ipynb",
        "kind": "chat",
        "summary": "Flash-Next (ISTA/Gyro/Swift) + Qwen3.6 + DeepSeek (strata-dsv41)",
    },
    "maya": {
        "title": "Maya",
        "notebook": "notebooks/maya/maya_hub_colab.ipynb",
        "kind": "chat",
        "summary": "GLM-5.3-Flash (Maya-S24/S/XS) + DeepSeek-V4.1-Flash-1017 (maya-ds4)",
    },
    "tensorfold": {
        "title": "TensorFold",
        "notebook": "notebooks/tensorfold/tensorfold_hub_colab.ipynb",
        "kind": "chat",
        "summary": "Qwen3.8-27B Mia EXL3 2.0bpw + DFlash2 (workstation qwen27b-tensorfold)",
    },
    "comfyui": {
        "title": "ComfyUI",
        "notebook": "notebooks/comfyui/comfyui_hub_colab.ipynb",
        "kind": "image",
        "summary": "Qwen-Image-2.1 Turbo, FLUX.3 Klein, Wan 2.2 / MiniMax-H3 video prefetch",
    },
    "sparsa": {
        "title": "Sparsa",
        "notebook": "notebooks/sparsa/sparsa_hub_colab.ipynb",
        "kind": "chat",
        "summary": "MiniMax-M3.1-Flash (IQ2_XS-D3 / IQ2_KT) via Sparsa strata-minimax",
    },
}
INDEX_NOTEBOOK = "notebooks/llm_hub_colab.ipynb"


def band(vram_gb):
    if vram_gb < 20:
        return "t4"
    if vram_gb < 30:
        return "l4"
    if vram_gb < 60:
        return "a100_40"
    if vram_gb < 90:
        return "a100_80"
    return "g4"


def for_engine(engine, vram_gb, entries=None):
    """Return {engine, band, engines, prefetch, notes, visible, skipped} for one engine notebook."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; choose from {ENGINES}")
    entries = entries if entries is not None else catalog.load()
    scoped = catalog.for_engines(entries, [engine])
    by = catalog.by_id(scoped)
    b = band(vram_gb)
    want = list(_ENGINE_PREFETCH[engine][b])
    prefetch = [m for m in want if m in by and catalog.fits(by[m], vram_gb)]
    return {
        "engine": engine,
        "band": b,
        "engines": [engine],
        "prefetch": prefetch,
        "notes": list(_ENGINE_NOTES[engine][b]),
        "visible": [e["id"] for e in catalog.visible(scoped, vram_gb)],
        "skipped": [m for m in want if m not in prefetch],
        "meta": dict(ENGINE_META[engine]),
    }


def for_gpu(vram_gb, entries=None):
    """Multi-engine overview (index notebook / docs). Prefer for_engine() in per-engine notebooks."""
    entries = entries if entries is not None else catalog.load()
    plans = [for_engine(eng, vram_gb, entries) for eng in ENGINES]
    prefetch, seen = [], set()
    for plan in plans:
        for mid in plan["prefetch"]:
            if mid not in seen:
                seen.add(mid)
                prefetch.append(mid)
    engines = [p["engine"] for p in plans if p["prefetch"] or p["band"] != "t4"
               or p["engine"] in ("diffusers", "comfyui")]
    if not engines:
        engines = ["diffusers"]
    notes = []
    for plan in plans:
        if plan["prefetch"] or plan["engine"] in ("diffusers", "comfyui"):
            notes.append(f"[{plan['engine']}] " + plan["notes"][0])
    return {
        "band": band(vram_gb),
        "engines": engines,
        "prefetch": prefetch,
        "notes": notes or ["No recommended models for this VRAM band."],
        "visible": [e["id"] for e in catalog.visible(entries, vram_gb)],
        "skipped": [],
        "by_engine": {p["engine"]: p for p in plans},
    }


def explain(vram_gb, entries=None, engine=None):
    if engine:
        plan = for_engine(engine, vram_gb, entries)
        lines = [
            f"Engine: {plan['engine']} ({ENGINE_META[engine]['title']})",
            f"GPU band: {plan['band']} ({vram_gb} GB reported)",
            f"Prefetch: {', '.join(plan['prefetch']) or '(none that fit)'}",
        ]
        if plan["skipped"]:
            lines.append(f"Skipped (missing or does not fit): {', '.join(plan['skipped'])}")
        lines += plan["notes"]
        return "\n".join(lines)
    plan = for_gpu(vram_gb, entries)
    lines = [
        f"GPU band: {plan['band']} ({vram_gb} GB reported)",
        f"Engines with picks: {', '.join(plan['engines'])}",
        f"Combined prefetch hint: {', '.join(plan['prefetch']) or '(none)'}",
    ]
    lines += plan["notes"]
    return "\n".join(lines)
