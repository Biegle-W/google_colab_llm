"""python -m colab_llm_hub serve ... : run the hub server."""
import argparse
import os
import signal
import sys

from . import catalog, gpu, hub, keyfile, install
from .store import ModelStore, model_disk_gb


def parse(argv=None):
    p = argparse.ArgumentParser(prog="colab_llm_hub")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--drive-dir", default="/content/drive/MyDrive/llm_models")
    s.add_argument("--local-dir", default="/content/models")
    s.add_argument("--log-dir", default="/content/hub_logs")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--key-file", help="API key file (created if missing); default: <drive-dir>/hub/api_key")
    s.add_argument("--llama-bin", default="/content/llama.cpp/build/bin/llama-server")
    s.add_argument("--vllm-bin", default="vllm")
    s.add_argument("--strata-bin", default="strata", help="Strata Terminal binary (from ensure_strata)")
    s.add_argument("--strata-data", default=None, help="Strata --data-dir (default: <drive>/builds/strata_data)")
    s.add_argument("--maya-root", default=None, help="Project Maya checkout (from ensure_maya)")
    s.add_argument("--maya-models", default=None, help="Maya --models-dir (default: <drive>/builds/maya_models)")
    s.add_argument("--tensorfold-bin", default=None, help="tensorfold-native binary (from ensure_tensorfold)")
    s.add_argument("--comfyui-root", default=None, help="ComfyUI checkout (from ensure_comfyui)")
    s.add_argument("--sparsa-root", default=None, help="Sparsa checkout (from ensure_sparsa)")
    s.add_argument("--hyperqwen-root", default=None, help="HyperQwen checkout with patched venv (from ensure_hyperqwen)")
    s.add_argument("--ninfer-bin", default=None, help="ninfer-serve binary (from ensure_ninfer)")
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
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
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
    key = keyfile.load_or_create(a.key_file or os.path.join(a.drive_dir, "hub", "api_key"))
    store = ModelStore(a.drive_dir, a.local_dir, token=os.environ.get(a.hf_token_env))
    overrides = dict(x.split("=", 1) for x in a.ctx)
    builds = os.path.join(a.drive_dir, "builds")
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
