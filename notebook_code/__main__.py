"""python -m notebook_code serve ... : run the hub server."""
import argparse
import os
import signal
import sys

from . import catalog, gpu, hub, keyfile, install
from .store import ModelStore, model_disk_gb


def parse(argv=None):
    p = argparse.ArgumentParser(prog="notebook_code")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--drive-dir", default="/content/drive/MyDrive/colab_model_server")
    s.add_argument("--local-dir", default="/content/models")
    s.add_argument("--log-dir", default="/content/hub_logs")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--key-file", help="API key file (created if missing); default: <drive-dir>/access/api_key")
    s.add_argument("--llama-bin", default="/content/llama.cpp/build/bin/llama-server")
    s.add_argument("--vllm-bin", default="vllm")
    s.add_argument("--strata-bin", default="strata", help="Strata Terminal binary (from ensure_strata)")
    s.add_argument("--strata-data", default=None, help="Strata --data-dir (default: <drive>/engines/strata_data)")
    s.add_argument("--maya-root", default=None, help="Project Maya checkout (from ensure_maya)")
    s.add_argument("--maya-models", default=None, help="Maya --models-dir (default: <drive>/engines/maya_models)")
    s.add_argument("--tensorfold-bin", default=None, help="tensorfold-native binary (from ensure_tensorfold)")
    s.add_argument("--comfyui-root", default=None, help="ComfyUI checkout (from ensure_comfyui)")
    s.add_argument("--sparsa-root", default=None, help="Sparsa checkout (from ensure_sparsa)")
    s.add_argument("--hyperqwen-root", default=None, help="HyperQwen checkout with patched venv (from ensure_hyperqwen)")
    s.add_argument("--ninfer-bin", default=None, help="ninfer-serve binary (from ensure_ninfer)")
    s.add_argument("--basalt-root", default=None, help="Basalt build (from ensure_basalt)")
    s.add_argument("--strata-nvfp4-root", default=None, help="Strata NVFP4 checkout (from ensure_strata_nvfp4)")
    s.add_argument("--python", default=sys.executable, help="Python used to launch Diffusers / ComfyUI / Sparsa")
    s.add_argument("--catalog", default=None)
    s.add_argument("--engines",
                   default=",".join(catalog.BACKENDS),
                   help="comma-separated engines")
    s.add_argument("--models", default="",
                   help="comma-separated model ids to serve (default: every catalog model that fits); "
                        "the notebooks pass only the selected model so nothing else is downloaded")
    s.add_argument("--vram-gb", type=float, default=None, help="override the detected VRAM (testing)")
    s.add_argument("--arch", default=None, help="override the GPU compute capability, e.g. 120 (testing)")
    s.add_argument("--ram-gb", type=float, default=None, help="override the detected system RAM (testing)")
    s.add_argument("--disk-gb", type=float, default=None,
                   help="override the local disk available for models (testing)")
    s.add_argument("--ctx", action="append", default=[], metavar="MODEL=TOKENS",
                   help="override a model's context tokens (chat) or max edge px (image)")
    s.add_argument("--hf-token-env", default="HF_TOKEN")
    r = sub.add_parser("prepare", help="download a model and build its engine on Drive for a GPU runtime, "
                                         "from any runtime (no GPU needed)")
    r.add_argument("--target", required=True, help="GPU runtime to prepare for: l4, a100 or g4")
    r.add_argument("--engine", default=None, help="engine to build (single-model mode)")
    r.add_argument("--models", default=None,
                   help="batch mode: comma-separated model ids, or recommended-chat / recommended-image / "
                        "recommended-all for the target's recommended picks; downloads only (no engine builds)")
    r.add_argument("--budget-gb", type=float, default=700, help="batch mode: stop after this much new data "
                   "(Google Drive accepts ~750 GB of uploads per day)")
    r.add_argument("--workers", type=int, default=1, help="batch mode: models downloaded at once")
    r.add_argument("--no-pace", action="store_true",
                   help="batch mode: do not wait for each file's Drive upload before copying the next")
    r.add_argument("--model", default="recommended", help="catalog id, or 'recommended' for the target")
    r.add_argument("--drive-dir", default="/content/drive/MyDrive/colab_model_server")
    r.add_argument("--staging-dir", default="/content/prepare_staging", help="local disk for files in transit")
    r.add_argument("--no-engine", action="store_true", help="only download the model")
    r.add_argument("--no-download", action="store_true", help="only build / cache the engine")
    r.add_argument("--refresh", action="store_true", help="rebuild the engine cache")
    r.add_argument("--force-compile", action="store_true",
                   help="compile llama.cpp / NInfer here even on a runtime with few CPU threads")
    r.add_argument("--hf-token-env", default="HF_TOKEN")
    return p.parse_args(argv)


def run_prepare(a):
    from . import prepare
    sys.stdout.reconfigure(line_buffering=True)         # progress shows up in a log file as it happens
    if a.models:
        if a.models.startswith("recommended-"):
            ids = prepare.recommended_models(a.target, kind=a.models.split("-", 1)[1])
        else:
            ids = [m for m in a.models.split(",") if m]
        from .drive_upload import upload_paced_copier
        result = prepare.prepare_many(ids, drive_dir=a.drive_dir, staging=a.staging_dir,
                                      token=os.environ.get(a.hf_token_env), budget_gb=a.budget_gb,
                                      workers=a.workers, copier=None if a.no_pace else upload_paced_copier())
        return 1 if result["errors"] else 0
    if not a.engine:
        raise SystemExit("prepare needs --engine (one model) or --models (batch)")
    result = prepare.prepare(a.target, a.engine, a.model, drive_dir=a.drive_dir, staging=a.staging_dir,
                             token=os.environ.get(a.hf_token_env), do_engine=not a.no_engine,
                             do_download=not a.no_download, refresh=a.refresh,
                             force_compile=a.force_compile)
    return 1 if result["errors"] else 0


def main(argv=None):
    a = parse(argv)
    if a.cmd == "prepare":
        return run_prepare(a)
    # Prefer Drive-backed logs / HF cache when the workspace module is available.
    try:
        from .workspace import open_workspace
        ws = open_workspace(a.drive_dir, a.local_dir)
        if a.log_dir == "/content/hub_logs":
            a.log_dir = str(ws.engine_logs)
    except Exception:
        ws = None
    entries = catalog.for_engines(catalog.load(a.catalog), a.engines.split(","))
    if a.models:
        entries = catalog.only(entries, a.models.split(","))
    if a.vram_gb is None:
        g = gpu.detect()[0]
        vram, name, arch = g["vram_gb"], g["name"], g["arch"]
        gpu.print_runtime(g)
    else:
        vram, name, arch = a.vram_gb, "override", a.arch
        for line in gpu.format_host():
            print(line)
    key = keyfile.load_or_create(a.key_file or os.path.join(a.drive_dir, "access", "api_key"))
    store = ModelStore(a.drive_dir, a.local_dir, token=os.environ.get(a.hf_token_env))
    overrides = dict(x.split("=", 1) for x in a.ctx)
    builds = os.path.join(a.drive_dir, "engines")
    bins = {
        "llama": a.llama_bin, "vllm": a.vllm_bin, "python": a.python,
        "strata": a.strata_bin,
        "strata_data": a.strata_data or os.path.join(builds, "strata_data"),
        "maya": a.maya_root or os.path.join(builds, "project-maya"),
        "maya_models": a.maya_models or os.path.join(builds, "maya_models"),
        "tensorfold": a.tensorfold_bin or os.path.join(
            builds, f"tensorfold-{install.TENSORFOLD_VERSION}", "bin", "tensorfold-native"),
        "comfyui": a.comfyui_root or os.path.join(builds, "ComfyUI"),
        "sparsa": a.sparsa_root or os.path.join(builds, "Sparsa"),
        "hyperqwen": a.hyperqwen_root or "/content/HyperQwen",
        "ninfer": a.ninfer_bin or "/content/ninfer/apps/ninfer-serve",
        "basalt": a.basalt_root or install.BASALT_ROOT,
        "strata_nvfp4": a.strata_nvfp4_root or install.STRATA_NVFP4_ROOT,
        "builds": builds,
        "work_dir": a.local_dir,            # engine scratch on local disk (models themselves stay on Drive)
    }
    ram = a.ram_gb if a.ram_gb is not None else gpu.ram()["total_gb"]
    disk = a.disk_gb if a.disk_gb is not None else model_disk_gb(a.local_dir)
    h = hub.Hub(entries, vram, store, bins, a.log_dir,
                ctx_overrides={k: int(v) for k, v in overrides.items()}, gpu_name=name,
                ram_gb=ram, disk_gb=disk, arch=arch)
    server = hub.make_server(h, key, "127.0.0.1", a.port)

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    print(f"Hub on 127.0.0.1:{a.port}; {name}, {vram} GB VRAM, {ram} GB RAM, {disk} GB disk for models; "
          f"{len(h.entries)} of {len(entries)} models fit.", flush=True)
    for e in entries:
        if e["id"] not in h.entries:
            print(f"  not served: {e['id']} ({catalog.why_not(e, vram, ram, disk, arch)})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        h.shutdown()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
