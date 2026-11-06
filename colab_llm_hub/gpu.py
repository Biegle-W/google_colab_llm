"""Runtime hardware detection: NVIDIA GPU, host CPU, and system RAM.

VRAM is reported in decimal GB so it compares directly with Hugging Face file sizes.
RAM uses the same decimal-GB convention (from /proc/meminfo kB).
"""
import os
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
    try:
        r = run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True)
    except OSError as e:
        raise RuntimeError(
            "No NVIDIA GPU found. Runtime -> Change runtime type -> choose an L4, A100 or G4 GPU."
        ) from e
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


def parse_cpuinfo(text, logical=None):
    """Parse /proc/cpuinfo into model name, physical cores, and logical threads."""
    model = None
    physical_ids = set()
    core_ids = set()
    processors = 0
    cur_phys = cur_core = None
    for line in (text or "").splitlines():
        if not line.strip():
            if cur_phys is not None and cur_core is not None:
                core_ids.add((cur_phys, cur_core))
            cur_phys = cur_core = None
            continue
        if ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        if key == "model name" and not model:
            model = val
        elif key == "processor":
            processors += 1
        elif key == "physical id":
            cur_phys = val
            physical_ids.add(val)
        elif key == "core id":
            cur_core = val
    if cur_phys is not None and cur_core is not None:
        core_ids.add((cur_phys, cur_core))
    threads = logical if logical is not None else (processors or os.cpu_count() or 1)
    cores = len(core_ids) if core_ids else threads
    return {
        "model": model or "unknown CPU",
        "cores": cores,
        "threads": threads,
        "sockets": max(1, len(physical_ids)) if physical_ids else 1,
    }


def parse_meminfo(text):
    """Parse /proc/meminfo into decimal GB totals (MemTotal / MemAvailable / derived used)."""
    vals = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.split()
        if not parts:
            continue
        try:
            vals[key] = int(parts[0])  # kB
        except ValueError:
            continue
    total_kb = vals.get("MemTotal", 0)
    avail_kb = vals.get("MemAvailable", vals.get("MemFree", 0))
    used_kb = max(0, total_kb - avail_kb)
    to_gb = lambda kb: round(kb / 1_000_000, 1)
    return {
        "total_gb": to_gb(total_kb),
        "available_gb": to_gb(avail_kb),
        "used_gb": to_gb(used_kb),
        "total_kb": total_kb,
        "available_kb": avail_kb,
    }


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def cpu(read=_read, logical=None):
    """Host CPU model and core/thread counts from /proc/cpuinfo."""
    try:
        text = read("/proc/cpuinfo")
    except OSError:
        text = ""
    return parse_cpuinfo(text, logical=logical if logical is not None else os.cpu_count())


def ram(read=_read):
    """Host RAM totals from /proc/meminfo (decimal GB)."""
    try:
        text = read("/proc/meminfo")
    except OSError:
        text = ""
    return parse_meminfo(text)


def host(read=_read, logical=None):
    """CPU + RAM snapshot for the current runtime."""
    return {"cpu": cpu(read=read, logical=logical), "ram": ram(read=read)}


def format_host(h=None, read=_read, logical=None):
    """One-line CPU and one-line RAM summary."""
    h = h or host(read=read, logical=logical)
    c, r = h["cpu"], h["ram"]
    socket_note = f" ({c['sockets']} sockets)" if c["sockets"] > 1 else ""
    return (
        f"CPU: {c['model']} — {c['cores']} cores / {c['threads']} threads{socket_note}",
        f"RAM: {r['total_gb']} GB total, {r['available_gb']} GB available "
        f"({r['used_gb']} GB used)",
    )


def print_runtime(gpu_info=None, run=subprocess.run, read=_read, logical=None, disk_path="/content"):
    """Print GPU (if given or detected), CPU, RAM, and free disk on every call."""
    g = gpu_info
    if g is None:
        try:
            g = detect(run=run)[0]
        except RuntimeError as e:
            print(f"GPU: {e}")
            g = None
    if g:
        print(f"GPU: {g['name']}: {g['vram_gb']} GB VRAM, compute capability "
              f"{g['compute_cap']} (sm_{g['arch']})")
    for line in format_host(read=read, logical=logical):
        print(line)
    for path in (disk_path, "/mnt/local-scratch"):        # the A100 80 GB adds a fast NVMe scratch disk
        if path != disk_path and not os.path.isdir(path):
            continue
        try:
            usage = os.statvfs(path)
            free_gb = round(usage.f_bavail * usage.f_frsize / 1_000_000_000, 1)
            total_gb = round(usage.f_blocks * usage.f_frsize / 1_000_000_000, 1)
            print(f"Disk ({path}): {free_gb} GB free / {total_gb} GB total")
        except OSError:
            pass
    return {"gpu": g, **host(read=read, logical=logical)}


def cmake_arch(arch, cuda_version):
    """CUDA architecture string for llama.cpp. Blackwell (sm_120) needs CUDA 12.8+; older toolkits fall back to JIT-able PTX."""
    if int(arch) >= 120 and tuple(cuda_version) < (12, 8):
        return "90-virtual"
    return str(arch)


def nvcc_version(text):
    m = re.search(r"release (\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None
