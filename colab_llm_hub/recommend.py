"""Pick catalog ids for one Colab engine (or a multi-engine overview).

Each hub notebook locks to a single engine. Recommendations follow ENGINE_MATRIX.md
(your NVIDIA/AMD registries + community reports). Strata / Maya / HyperQwen-container
are documented there but not launched from Colab.
"""
from . import catalog

ENGINES = ("llama", "vllm", "sglang", "diffusers")

# Preferred catalog ids per engine × GPU band (best-first).
_ENGINE_PREFETCH = {
    "llama": {
        "t4": [],
        "l4": ["qwen3.8-27b-q4"],
        "a100_40": ["qwen3.8-27b-q8", "flash-next-iq1s", "glm-5.3-flash-iq1s"],
        "a100_80": ["qwen3.8-27b-q8", "flash-next-q3", "glm-5.3-flash-iq1s"],
        "g4": ["flash-next-q3", "glm-5.3-flash-iq1s", "deepseek-v4.1-flash-1017-iq1s",
               "qwen3.8-27b-bf16"],
    },
    "vllm": {
        "t4": [],
        "l4": ["qwen3.8-27b-w4a16", "qwen3-14b-fp8"],
        "a100_40": ["qwen3.8-27b-fp8", "qwen3.8-27b-w4a16", "qwen3.6-35b-a3b-fp8"],
        "a100_80": ["qwen3.8-27b-fp8", "sage-a1.5-fp8", "qwen3.8-27b-bf16-vllm"],
        "g4": ["qwen3.8-27b-fp8", "qwen3.8-27b-bf16-vllm", "sage-a1.5-fp8"],
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
}

_ENGINE_NOTES = {
    "llama": {
        "t4": [
            "T4: llama.cpp chat targets need L4+ (AI Pro / Colab Pro).",
            "Use the Diffusers notebook for NF4 image models on a free T4.",
        ],
        "l4": [
            "L4 (24 GB): UD-Q4 Qwen3.8-27B is the practical llama.cpp chat pick.",
            "Flash-Next / GLM-5.3 / DeepSeek-V4.1 need A100 high-RAM or G4 (+ host RAM).",
        ],
        "a100_40": [
            "A100 40 GB (+ ~80 GB host): Flash-Next IQ1_S; GLM-5.3-Flash IQ1_S (experimental).",
            "Needs a llama.cpp build with GLM / large-MoE support when loading those GGUFs.",
        ],
        "a100_80": [
            "A100 80 GB: Flash-Next Q3; GLM-5.3-Flash IQ1_S.",
            "Add deepseek-v4.1-flash-1017-iq1s manually if you accept a ~233 GB Drive download.",
        ],
        "g4": [
            "G4 (~96 GB): best Colab band for large MoE GGUFs.",
            "Flash-Next Q3; GLM-5.3 IQ1_S; DeepSeek-V4.1-Flash-1017 IQ1_S (Engram mmap).",
            "Workstation still prefers Maya (GLM) and Maya/Strata (DeepSeek).",
        ],
    },
    "vllm": {
        "t4": [
            "T4: vLLM targets need L4+.",
        ],
        "l4": [
            "L4 (24 GB): W4A16 AutoRound Qwen3.8-27B (HyperQwen-compatible weights) or Qwen3-14B FP8.",
            "Official Flash-Next / GLM / DeepSeek vLLM recipes are multi-GPU — not in this notebook.",
        ],
        "a100_40": [
            "A100 40 GB: Qwen3.8-27B FP8 or W4A16; Qwen3.6-35B-A3B FP8 when it fits.",
        ],
        "a100_80": [
            "A100 80 GB: FP8 / BF16 dense chat; Sage A1.5 FP8.",
        ],
        "g4": [
            "G4: Qwen3.8-27B FP8 or BF16; Sage A1.5 FP8.",
        ],
    },
    "sglang": {
        "t4": [
            "T4: SGLang targets need L4+.",
        ],
        "l4": [
            "L4+: Qwen3.8-27B W4A16 via SGLang (same AutoRound weights as the vLLM entry).",
            "Flash-Next / GLM / DeepSeek official SGLang paths want multi-GPU — use the llama.cpp notebook.",
        ],
        "a100_40": [
            "A100: SGLang W4A16 for Qwen3.8-27B. Add more SGLang catalog ids as you enable them.",
        ],
        "a100_80": [
            "A100 80 GB / G4: same W4A16 chat entry; higher ctx room.",
        ],
        "g4": [
            "G4: Qwen3.8-27B W4A16 via SGLang.",
        ],
    },
    "diffusers": {
        "t4": [
            "T4: Qwen-Image-2.1 NF4 or SD 1.5.",
            "MiniMax-H3 is video — prefetch only; run via ComfyUI / SGLang diffusion elsewhere.",
        ],
        "l4": [
            "L4: Qwen-Image-2.1 NF4, SDXL Turbo / base.",
        ],
        "a100_40": [
            "A100 40 GB: Qwen-Image-2.1 BF16, FLUX.1-schnell.",
        ],
        "a100_80": [
            "A100 80 GB: Qwen-Image-2.1 BF16, FLUX.1-schnell.",
        ],
        "g4": [
            "G4: Qwen-Image-2.1 BF16, FLUX; MiniMax-H3 prefetch for ComfyUI (experimental).",
        ],
    },
}

ENGINE_META = {
    "llama": {
        "title": "llama.cpp",
        "notebook": "llama_hub_colab.ipynb",
        "kind": "chat",
        "summary": "GGUF chat: Qwen3.8-27B, Flash-Next, GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017",
    },
    "vllm": {
        "title": "vLLM",
        "notebook": "vllm_hub_colab.ipynb",
        "kind": "chat",
        "summary": "Safetensors chat: Qwen3.8-27B W4A16/FP8/BF16, Sage, Nemotron Bolt",
    },
    "sglang": {
        "title": "SGLang",
        "notebook": "sglang_hub_colab.ipynb",
        "kind": "chat",
        "summary": "SGLang chat: Qwen3.8-27B W4A16 AutoRound",
    },
    "diffusers": {
        "title": "Diffusers",
        "notebook": "diffusers_hub_colab.ipynb",
        "kind": "image",
        "summary": "Text-to-image: Qwen-Image-2.1, FLUX, SDXL (+ MiniMax-H3 prefetch notes)",
    },
}


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
    # Prefer chat engines then images for a combined prefetch hint.
    prefetch, seen = [], set()
    for plan in plans:
        for mid in plan["prefetch"]:
            if mid not in seen:
                seen.add(mid)
                prefetch.append(mid)
    engines = [p["engine"] for p in plans if p["prefetch"] or p["band"] != "t4" or p["engine"] == "diffusers"]
    if not engines:
        engines = ["diffusers"]
    notes = []
    for plan in plans:
        if plan["prefetch"] or plan["engine"] == "diffusers":
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
