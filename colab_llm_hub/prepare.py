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


def recommended_models(target, kind="chat", entries=None, skip_engine_managed=True):
    """The catalog's recommended models for a target runtime, best first, across all engines."""
    entries = entries if entries is not None else catalog.load()
    by = catalog.by_id(entries)
    vram, ram, disk, arch = runtimes.specs(target)
    out = []
    for engine in recommend.ENGINES:
        for m in recommend.for_engine(engine, vram, entries, ram_gb=ram, disk_gb=disk, arch=arch)["prefetch"]:
            k = catalog.kind_for(by[m])
            if m in out or (kind == "chat" and k != "chat") or (kind == "image" and k == "chat"):
                continue
            if skip_engine_managed and by[m].get("engine_download"):
                continue
            out.append(m)
    return out


def free_gb(path):
    import shutil
    return shutil.disk_usage(path).free / 1e9


def prepare_many(models, drive_dir="/content/drive/MyDrive/llm_models", staging=DEFAULT_STAGING, token=None,
                 budget_gb=700.0, min_free_gb=40.0, workers=1, wait_free_s=1800, log=print, store=None,
                 free=free_gb, sleep=None, entries=None, copier=None):
    """Save several models to Drive, `workers` at a time, until `budget_gb` of new data (Google Drive takes
    about 750 GB of uploads per day per account). Models already complete on Drive are skipped, so running
    it again the next day continues where it stopped.

    Colab's Drive client keeps what it uploads in a local cache, so before each model it waits (up to
    `wait_free_s`) for at least `min_free_gb` + room for two of the model's largest files on local disk,
    and stops cleanly instead of filling the disk. Returns {"saved", "skipped", "deferred", "errors"}."""
    import concurrent.futures
    import time as _time
    sleep = sleep or _time.sleep
    entries = entries if entries is not None else catalog.load()
    by = catalog.by_id(entries)
    unknown = [m for m in models if m not in by]
    if unknown:
        raise ValueError(f"unknown model ids: {', '.join(unknown)}")
    store = store or ModelStore(drive_dir, staging, token=token, log=log)
    result = {"saved": [], "skipped": [], "deferred": [], "errors": {}}
    queue, planned = [], 0.0
    for m in models:
        e = by[m]
        if store.state(e)["on_drive"]:
            result["skipped"].append(m)
        elif e.get("engine_download"):
            result["errors"][m] = "downloaded by its engine on first start; not saved from here"
        elif planned + e["size_gb"] > budget_gb and queue:
            result["deferred"].append(m)
        else:
            queue.append(m)
            planned += e["size_gb"]
    log(f"[prepare] {len(queue)} models, {planned:.0f} GB this run; {len(result['skipped'])} already on Drive; "
        f"{len(result['deferred'])} left for the next run (budget {budget_gb:.0f} GB)")
    stop = threading.Event()
    lock = threading.Lock()

    def staged_gb(m):
        """Data an earlier, interrupted run already left in this model's staging folder (it is reused)."""
        total = 0
        for root, _, files in os.walk(os.path.join(staging, m)):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total / 1e9

    def room_for(e):
        # Two of the model's largest files in flight (<= 50 GB each), minus what is already staged for it.
        need = min_free_gb + max(0.0, 2 * min(float(e["size_gb"]), 50.0) - staged_gb(e["id"]))
        waited = 0
        while free(staging if os.path.isdir(staging) else "/") < need:
            if waited >= wait_free_s or stop.is_set():
                return False
            if waited == 0:
                log(f"[prepare] waiting for local disk (Drive cache) to drop below the limit before {e['id']}")
            sleep(30)
            waited += 30
        return True

    def one(m):
        e = by[m]
        if stop.is_set():
            with lock:
                result["deferred"].append(m)
            return
        if not room_for(e):
            stop.set()
            with lock:
                result["deferred"].append(m)
                result["errors"][m] = "local disk stayed full (Drive cache); run again later"
            return
        try:
            if copier is not None:
                store.download_to_drive(e, staging=os.path.join(staging, m), copier=copier)
            else:
                store.download_to_drive(e, staging=os.path.join(staging, m))
            with lock:
                result["saved"].append(m)
        except Exception as ex:
            if type(ex).__name__ == "UploadStalled":         # never queue more behind a stuck Drive client
                stop.set()
            with lock:
                result["errors"][m] = f"{type(ex).__name__}: {ex}"
            log(f"[prepare] {m} failed: {ex}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, queue))
    log(f"[prepare] saved {len(result['saved'])}, skipped {len(result['skipped'])}, "
        f"deferred {len(result['deferred'])}, failed {len(result['errors'])}")
    return result
