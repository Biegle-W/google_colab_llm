"""Pick engines and catalog ids for the four target models on a Colab GPU.

Recommendations follow ENGINE_MATRIX.md (your NVIDIA/AMD registries + community reports).
Strata / Maya / HyperQwen-container are documented there but not launched from Colab.
"""
from . import catalog

# Preferred catalog ids per model family, ordered best-first for each VRAM band.
TARGET_FAMILIES = ("qwen3.8-27b", "flash-next", "qwen-image-2.1", "minimax-h3")


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
            "T4: only Qwen-Image-2.1 NF4 is practical among the four targets.",
            "Qwen3.8-27B / Flash-Next / MiniMax-H3 need L4+ (AI Pro / Colab Pro).",
        ],
    },
    "l4": {
        "engines": ["llama", "vllm", "diffusers"],
        "prefetch": ["qwen3.8-27b-w4a16", "qwen3.8-27b-q4", "qwen-image-2.1-nf4"],
        "notes": [
            "L4 (24 GB): vLLM W4A16 for Qwen3.8-27B (same AutoRound class as HyperQwen); "
            "llama.cpp Q4 as fallback; Qwen-Image NF4 or BF16 if VRAM allows.",
            "Flash-Next and MiniMax-H3 need more combined RAM/VRAM than a typical L4 VM.",
            "Optional: add 'sglang' and prefetch qwen3.8-27b-w4a16-sglang for SGLang.",
        ],
    },
    "a100_40": {
        "engines": ["llama", "vllm", "sglang", "diffusers"],
        "prefetch": ["qwen3.8-27b-fp8", "qwen3.8-27b-w4a16-sglang", "flash-next-iq1s", "qwen-image-2.1"],
        "notes": [
            "A100 40 GB: vLLM FP8 or SGLang W4A16 for 27B; Flash-Next IQ1_S via llama.cpp "
            "with n-gram on CPU (needs the ~80 GB host RAM Colab often gives with A100).",
            "Qwen-Image-2.1 BF16. MiniMax-H3 remains experimental (ComfyUI / SGLang diffusion offload).",
            "Workstation Strata/Maya for Flash-Next are not Colab-portable.",
        ],
    },
    "a100_80": {
        "engines": ["llama", "vllm", "sglang", "diffusers"],
        "prefetch": ["qwen3.8-27b-fp8", "flash-next-q3", "qwen-image-2.1", "minimax-h3"],
        "notes": [
            "A100 80 GB: vLLM FP8 for 27B; Flash-Next Q3 GGUF with n-gram CPU offload; "
            "Qwen-Image BF16; MiniMax-H3 experimental with heavy offload.",
        ],
    },
    "g4": {
        "engines": ["llama", "vllm", "sglang", "diffusers"],
        "prefetch": ["qwen3.8-27b-fp8", "flash-next-q3", "qwen-image-2.1", "minimax-h3"],
        "notes": [
            "G4 (~96 GB): best single-GPU Colab for Flash-Next (llama.cpp UD-Q3_K_XL + n-gram CPU; "
            "community ~63 GB VRAM @ 131K). vLLM FP8/NVFP4 or SGLang for 27B. Qwen-Image BF16.",
            "MiniMax-H3: ComfyUI or SGLang diffusion still preferred over fitting full BF16 resident.",
        ],
    },
}


def for_gpu(vram_gb, entries=None):
    """Return {band, engines, prefetch, notes, visible} for this GPU."""
    entries = entries if entries is not None else catalog.load()
    by = catalog.by_id(entries)
    plan = _PLANS[band(vram_gb)]
    prefetch = [m for m in plan["prefetch"] if m in by and catalog.fits(by[m], vram_gb)]
    # Drop experimental video unless explicitly still listed and fitting.
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
