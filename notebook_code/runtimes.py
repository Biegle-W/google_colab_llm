"""The Colab GPU runtimes models and engines are prepared for, measured on real sessions.

A cheap CPU runtime uses these to check fit and build for a GPU it does not have: downloads and
engine builds go to Drive, and the GPU runtime restores them.
"""

# vram_gb: nvidia-smi MiB in decimal GB. disk_gb: free space where models go (A100: /mnt/local-scratch NVMe;
# L4 / G4: /content), with some margin. arch: compute capability without the dot.
TARGETS = {
    "l4": {"name": "L4 (RTX 4070 Ti class)", "vram_gb": 24.2, "arch": "89", "ram_gb": 55.5, "disk_gb": 225},
    "a100": {"name": "A100 80 GB", "vram_gb": 85.9, "arch": "80", "ram_gb": 175.2, "disk_gb": 340},
    "g4": {"name": "G4 (RTX PRO 6000 Blackwell)", "vram_gb": 102.6, "arch": "120", "ram_gb": 185.5,
           "disk_gb": 253},   # /content free on a fresh G4 (measured)
}


def target(name):
    try:
        return TARGETS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown target runtime {name!r}; choose from {', '.join(TARGETS)}") from None


def specs(name):
    """(vram_gb, ram_gb, disk_gb, arch) in the order catalog.why_not takes them."""
    t = target(name)
    return t["vram_gb"], t["ram_gb"], t["disk_gb"], t["arch"]
