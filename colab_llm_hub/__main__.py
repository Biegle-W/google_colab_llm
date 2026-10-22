"""python -m colab_llm_hub serve ... : run the hub server."""
import argparse
import os
import signal
import sys

from . import catalog, gpu, hub, keyfile
from .store import ModelStore


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
    s.add_argument("--catalog", default=None)
    s.add_argument("--engines", default="llama,vllm", help="comma-separated engines that are installed")
    s.add_argument("--vram-gb", type=float, default=None, help="override the detected VRAM (testing)")
    s.add_argument("--ctx", action="append", default=[], metavar="MODEL=TOKENS", help="override a model's context")
    s.add_argument("--hf-token-env", default="HF_TOKEN")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    entries = catalog.for_engines(catalog.load(a.catalog), a.engines.split(","))
    if a.vram_gb is None:
        g = gpu.detect()[0]
        vram, name = g["vram_gb"], g["name"]
    else:
        vram, name = a.vram_gb, "override"
    key = keyfile.load_or_create(a.key_file or os.path.join(a.drive_dir, "hub", "api_key"))
    store = ModelStore(a.drive_dir, a.local_dir, token=os.environ.get(a.hf_token_env))
    overrides = dict(x.split("=", 1) for x in a.ctx)
    h = hub.Hub(entries, vram, store, {"llama": a.llama_bin, "vllm": a.vllm_bin}, a.log_dir,
                ctx_overrides={k: int(v) for k, v in overrides.items()}, gpu_name=name)
    server = hub.make_server(h, key, "127.0.0.1", a.port)

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    print(f"Hub on 127.0.0.1:{a.port}; {name}, {vram} GB; {len(h.entries)} of {len(entries)} models fit.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        h.shutdown()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
