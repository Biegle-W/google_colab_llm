"""Pick engines and catalog ids for the target models on a Colab GPU.

Recommendations follow ENGINE_MATRIX.md (your NVIDIA/AMD registries + community reports).
Strata / Maya / HyperQwen-container are documented there but not launched from Colab.
"""
from . import catalog

# Preferred catalog ids per model family, ordered best-first for each VRAM band.
TARGET_FAMILIES = (
    "qwen3.8-27b", "flash-next", "glm-5.3-flash", "deepseek-v4.1-flash-1017",
    "qwen-image-2.1", "minimax-h3",
)


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


# (engines_to_install, preferred_model_ids, notes)
_PLANS = {
    "t4": {
        "engines": ["diffusers"],
        "prefetch": ["qwen-image-2.1-nf4"],
        "notes": [
            "T4: only Qwen-Image-2.1 NF4 is practical among the targets.",
            "LLMs (27B / Flash-Next / GLM-5.3 / DeepSeek-V4.1) need L4+ (AI Pro / Colab Pro).",
        ],
    },
    "l4": {
        "engines": ["llama", "vllm", "diffusers"],
        "prefetch": ["qwen3.8-27b-w4a16", "qwen3.8-27b-q4", "qwen-image-2.1-nf4"],
        "notes": [
            "L4 (24 GB): vLLM W4A16 for Qwen3.8-27B (same AutoRound class as HyperQwen); "
            "llama.cpp Q4 as fallback; Qwen-Image NF4.",
            "GLM-5.3-Flash (~93 GB+) and DeepSeek-V4.1-Flash-1017 (~233 GB) need far more "
            "combined RAM/VRAM than an L4 VM — use A100 high-RAM or G4, or your Maya/Strata workstation.",
            "Optional: add 'sglang' and prefetch qwen3.8-27b-w4a16-sglang for SGLang.",
        ],
    },
    "a100_40": {
        "engines": ["llama", "vllm", "sglang", "diffusers"],
        "prefetch": ["qwen3.8-27b-fp8", "qwen3.8-27b-w4a16-sglang", "flash-next-iq1s",
                     "glm-5.3-flash-iq1s", "qwen-image-2.1"],
        "notes": [
            "A100 40 GB (+ ~80 GB host): vLLM FP8 / SGLang for 27B; Flash-Next IQ1_S; "
            "GLM-5.3-Flash IQ1_S is experimental (needs ~100 GB combined + GLM-capable llama.cpp).",
            "DeepSeek-V4.1-Flash-1017 IQ1_S (~233 GB on Drive) is usually too large for this VM; "
            "use G4 with mmap or your Maya/Strata workstation.",
            "Workstation: GLM → Maya-S24; DeepSeek → Maya / Strata (not Colab-portable).",
        ],
    },
    "a100_80": {
        "engines": ["llama", "vllm", "sglang", "diffusers"],
        "prefetch": ["qwen3.8-27b-fp8", "flash-next-q3", "glm-5.3-flash-iq1s", "qwen-image-2.1"],
        "notes": [
            "A100 80 GB: vLLM FP8 for 27B; Flash-Next Q3; GLM-5.3-Flash IQ1_S (experimental). "
            "Add deepseek-v4.1-flash-1017-iq1s manually if you accept a ~233 GB Drive download + Engram mmap.",
        ],
    },
    "g4": {
        "engines": ["llama", "vllm", "sglang", "diffusers"],
        "prefetch": ["qwen3.8-27b-fp8", "flash-next-q3", "glm-5.3-flash-iq1s",
                     "deepseek-v4.1-flash-1017-iq1s", "qwen-image-2.1"],
        "notes": [
            "G4 (~96 GB): best single-GPU Colab band for large MoEs. Flash-Next Q3; GLM-5.3 IQ1_S; "
            "DeepSeek-V4.1-Flash-1017 IQ1_S (~233 GB Drive, Engram mmap/lazy — Unsloth llama.cpp).",
            "Your workstation still prefers Maya (GLM) and Maya/Strata (DeepSeek) when dual-GPU SSD streaming is available.",
            "Qwen3.8-27B: vLLM FP8/NVFP4 or SGLang. Qwen-Image BF16.",
        ],
    },
}


def for_gpu(vram_gb, entries=None):
    """Return {band, engines, prefetch, notes, visible} for this GPU."""
    entries = entries if entries is not None else catalog.load()
    by = catalog.by_id(entries)
    plan = _PLANS[band(vram_gb)]
    prefetch = [m for m in plan["prefetch"] if m in by and catalog.fits(by[m], vram_gb)]
    return {
        "band": band(vram_gb),
        "engines": list(plan["engines"]),
        "prefetch": prefetch,
        "notes": list(plan["notes"]),
        "visible": [e["id"] for e in catalog.visible(entries, vram_gb)],
        "skipped": [m for m in plan["prefetch"] if m not in prefetch],
    }


def explain(vram_gb, entries=None):
    plan = for_gpu(vram_gb, entries)
    lines = [f"GPU band: {plan['band']} ({vram_gb} GB reported)",
             f"Install engines: {', '.join(plan['engines'])}",
             f"Prefetch: {', '.join(plan['prefetch']) or '(none that fit)'}" ]
    if plan["skipped"]:
        lines.append(f"Skipped (missing or does not fit): {', '.join(plan['skipped'])}")
    lines += plan["notes"]
    return "\n".join(lines)
