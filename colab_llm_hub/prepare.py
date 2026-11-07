"""Get a model and its engine ready on Drive from a cheap CPU runtime, for a GPU runtime to use later.

Nothing here needs a GPU: downloads are network work, wheels and HyperQwen's venv are the same with or
without one, and nvcc compiles llama.cpp / NInfer for a named GPU it cannot see. The GPU notebook then
restores everything from Drive and spends its time on loading and inference.

    python -m colab_llm_hub prepare --target g4 --engine ninfer --model recommended
"""
import os
import threading

from . import catalog, install, recommend, runtimes
from .store import ModelStore, StoreError

DEFAULT_STAGING = "/content/prepare_staging"
# Engines compiled from source. A standard Colab CPU runtime has 2 threads and no CUDA toolkit, where these
# take far longer than the few minutes they take on a GPU runtime's many cores, so they are skipped there.
COMPILED_ENGINES = ("llama", "ninfer")
MIN_COMPILE_THREADS = 8


class Skipped(Exception):
    """The engine is better built on the GPU runtime; not an error."""


def pick(engine, target, model="recommended", entries=None):
    """The catalog entry to prepare, checked against the target runtime (not the one running this)."""
    entries = entries if entries is not None else catalog.load()
    vram, ram, disk, arch = runtimes.specs(target)
    scoped = catalog.by_id(catalog.for_engines(entries, [engine]))
    if model in (None, "", "recommended"):
        plan = recommend.for_engine(engine, vram, entries, ram_gb=ram, disk_gb=disk, arch=arch)
        fitting = [m for m, e in scoped.items() if catalog.fits(e, vram, ram, disk, arch)]
        model = (plan["prefetch"] or fitting or [None])[0]
        if model is None:
            raise ValueError(f"no {engine} model runs on the {runtimes.target(target)['name']} runtime")
    if model not in scoped:
        raise ValueError(f"{model} is not a {engine} model")
    reason = catalog.why_not(scoped[model], vram, ram, disk, arch)
    if reason:
        raise ValueError(f"{model} will not run on the {runtimes.target(target)['name']} runtime: {reason}")
    return scoped[model]


def build_engine(engine, target, builds_dir, refresh=False, run=install.sh, log=print, force_compile=False,
                 threads=None):
    """Put `engine`'s build (or wheels) for `target` on Drive. Returns what was cached."""
    arch = runtimes.target(target)["arch"]
    threads = threads or os.cpu_count() or 1
    if engine in COMPILED_ENGINES and threads < MIN_COMPILE_THREADS and not force_compile:
        raise Skipped(f"{engine} is compiled from source and this runtime has {threads} CPU threads; the GPU "
                      f"notebook builds it in a few minutes and caches it on Drive (force_compile to build here)")
    kw = dict(refresh=refresh, run=run, log=log)
    if engine == "llama":
        return install.ensure_llama(builds_dir, arch, rebuild=refresh, run=run, log=log, check=False)
    if engine == "vllm":
        return install.ensure_vllm(builds_dir, install_now=False, **kw)
    if engine == "sglang":
        return install.ensure_sglang(builds_dir, install_now=False, **kw)
    if engine == "diffusers":
        return install.ensure_diffusers(builds_dir, install_now=False, **kw)
    if engine == "hyperqwen":
        return install.ensure_hyperqwen(builds_dir, install_now=False, **kw)
    if engine == "ninfer":
        return install.ensure_ninfer(builds_dir, arch, install_now=False, **kw)
    ensure = {"strata": install.ensure_strata, "maya": install.ensure_maya, "tensorfold": install.ensure_tensorfold,
              "comfyui": install.ensure_comfyui, "sparsa": install.ensure_sparsa}.get(engine)
    if ensure is None:
        raise ValueError(f"unknown engine {engine!r}")
    return ensure(builds_dir, **kw)            # release downloads and git clones onto Drive


def prepare(target, engine, model="recommended", drive_dir="/content/drive/MyDrive/llm_models",
            staging=DEFAULT_STAGING, token=None, do_engine=True, do_download=True, refresh=False,
            log=print, store=None, engine_builder=None, force_compile=False):
    """Download the model to Drive while the engine builds; both run at once (network vs CPU).

    Returns {"target", "engine", "model", "engine_cache", "drive_path", "errors"}."""
    entry = pick(engine, target, model)
    t = runtimes.target(target)
    log(f"Preparing {entry['id']} ({entry['size_gb']:.1f} GB) and {engine} for the {t['name']} runtime "
        f"(sm_{t['arch']})")
    result = {"target": target, "engine": engine, "model": entry["id"], "engine_cache": None,
              "drive_path": None, "errors": {}, "skipped": {}}

    def download():
        try:
            s = store or ModelStore(drive_dir, staging, token=token, log=log)
            result["drive_path"] = str(s.download_to_drive(entry))
        except StoreError as e:
            result["errors"]["download"] = str(e)
        except Exception as e:                      # keep the engine build going; reported at the end
            result["errors"]["download"] = f"{type(e).__name__}: {e}"
    thread = threading.Thread(target=download, daemon=True) if do_download else None
    if thread:
        thread.start()
    if do_engine:
        try:
            builds = os.path.join(drive_dir, "builds")
            if engine_builder:
                result["engine_cache"] = engine_builder(engine, target, builds, refresh=refresh, log=log)
            else:
                result["engine_cache"] = build_engine(engine, target, builds, refresh=refresh, log=log,
                                                      force_compile=force_compile)
        except Skipped as e:
            result["skipped"]["engine"] = str(e)
            log(f"[prepare] engine skipped: {e}")
        except Exception as e:
            result["errors"]["engine"] = f"{type(e).__name__}: {e}"
    if thread:
        thread.join()
    for step, err in result["errors"].items():
        log(f"[prepare] {step} failed: {err}")
    if not result["errors"]:
        log(f"[prepare] done: open notebooks/{engine}/{engine}_gpu.ipynb on the {t['name']} runtime and pick "
            f"{entry['id']}; it restores both from Drive.")
    return result
