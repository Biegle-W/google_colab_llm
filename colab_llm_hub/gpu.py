"""NVIDIA GPU detection. VRAM is reported in decimal GB so it compares directly with Hugging Face file sizes."""
import re
import subprocess

QUERY = "name,memory.total,compute_cap"


def parse_smi(text):
    """Parse `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mib, cc = parts[0], float(parts[1]), parts[2]
        gpus.append({"name": name, "mem_mib": int(mib), "compute_cap": cc,
                     "arch": cc.replace(".", ""), "vram_gb": round(mib * 1.048576 / 1000, 1)})
    return gpus


def detect(run=subprocess.run):
    r = run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        raise RuntimeError("No NVIDIA GPU found. Runtime -> Change runtime type -> choose an L4, A100 or G4 GPU.")
    return parse_smi(r.stdout)


def used_mib(run=subprocess.run):
    """Memory currently in use on the first GPU, in MiB (None when nvidia-smi is unavailable)."""
    try:
        r = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20)
        return int(float(r.stdout.strip().splitlines()[0])) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
        return None


def cmake_arch(arch, cuda_version):
    """CUDA architecture string for llama.cpp. Blackwell (sm_120) needs CUDA 12.8+; older toolkits fall back to JIT-able PTX."""
    if int(arch) >= 120 and tuple(cuda_version) < (12, 8):
        return "90-virtual"
    return str(arch)


def nvcc_version(text):
    m = re.search(r"release (\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None
