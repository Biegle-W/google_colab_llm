"""Start HyperQwen for the hub: prepare the checkpoint once, then exec HyperQwen's own launcher.

HyperQwen (github.com/syv-ai/HyperQwen) expects its checkpoint to be post-processed by CPU-only scripts
(int8 lm_head / embeddings / MTP head, a 40k-token draft vocabulary). Those scripts rewrite files in place,
so they run on a local copy (<local models>/.hyperqwen/<id>) and never touch the hub's downloaded copy,
which may still be streaming to Drive. The copy is reused while its marker matches the source.

    python -m colab_llm_hub.hyperqwen_launch --root /content/HyperQwen --model /content/models/<id> \\
        --port 30030 --ctx 65536 --mode single --spec mtp
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PREPARE = (
    ("prepare/quant_lm_head.py", "{model}"),
    ("prepare/quant_embed.py", "{model}"),
    ("prepare/quant_mtp.py", "{model}"),
    ("prepare/build_draft_vocab.py", "{model}", "--ids", "prepare/draft_vocab_ids.json"),
)
MARKER = ".hyperqwen_prepared.json"
SHORT_CTX = 65536          # HyperQwen's CTX=fast profile (bf16 KV); above it CTX=long (fp8 KV)
# Qwen3.8 thinks by default and HyperQwen keeps that; like the reference stack, the hub turns it off unless a
# catalog entry asks for it. A client can still send chat_template_kwargs {"enable_thinking": true}.
NO_THINKING = '--default-chat-template-kwargs={"enable_thinking":false}'


def source_stamp(src):
    """What the prepared copy was made from: the hub's completion marker, else file names + sizes."""
    marker = Path(src) / ".complete.json"
    if marker.is_file():
        return marker.read_text()
    return json.dumps(sorted((p.name, p.stat().st_size) for p in Path(src).iterdir() if p.is_file()))


def prepared_dir(src, work_dir=None):
    """Local copy the prepare scripts rewrite. With the model read from Drive, work_dir (the runtime's
    local model disk) keeps that 20 GB copy off Drive."""
    src = Path(src)
    return Path(work_dir or src.parent) / ".hyperqwen" / src.name


def prepare(root, src, python, run=subprocess.run, log=print, work_dir=None):
    """Return a prepared copy of `src` (a downloaded dbirks/Qwen3.8-27B-W4A16-AutoRound directory)."""
    dest = prepared_dir(src, work_dir)
    stamp = source_stamp(src)
    marker = dest / MARKER
    if marker.is_file() and json.loads(marker.read_text()).get("source") == stamp:
        log(f"[hyperqwen] reusing prepared checkpoint {dest}")
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"[hyperqwen] copying {src} -> {dest} for HyperQwen's prepare scripts")
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".complete.json", "dflash2"))
    for script, *args in PREPARE:
        cmd = [python, script] + [a.format(model=str(dest)) for a in args]
        log("[hyperqwen] " + " ".join(cmd))
        r = run(cmd, cwd=str(root))
        if getattr(r, "returncode", 0):
            raise SystemExit(f"[hyperqwen] {script} failed (exit {r.returncode}); the copy in {dest} is incomplete")
    marker.write_text(json.dumps({"source": stamp}))
    return dest


def launch_env(model, port, ctx, mode, spec, drafter=None, base=None, thinking=False):
    env = dict(os.environ if base is None else base)
    env.pop("VLLM_API_KEY", None)        # no key -> HyperQwen binds 127.0.0.1; the hub does the auth
    env.update(MODEL=str(model), PORT=str(port), MAX_LEN=str(int(ctx)), HF_HUB_OFFLINE="1", HOST="127.0.0.1")
    if not thinking:     # HyperQwen splits EXTRA_ARGS on spaces, so the JSON carries none
        env["EXTRA_ARGS"] = (env.get("EXTRA_ARGS", "") + " " + NO_THINKING).strip()
    if mode == "single":
        env["SPEC"] = spec
        env["CTX"] = "fast" if int(ctx) <= SHORT_CTX else "long"
        if spec == "dflash2":
            if not drafter or not (Path(drafter) / "model.safetensors").is_file():
                raise SystemExit("[hyperqwen] SPEC=dflash2 needs the DFlash2 drafter in <model>/dflash2")
            env["DRAFT"] = str(drafter)
            env["DFLASH_MAX_LEN"] = str(int(ctx))
    return env


def script_for(root, mode):
    return str(Path(root) / ("batch" if mode == "batch" else "single-user") / "start_qwen.sh")


def main(argv=None):
    p = argparse.ArgumentParser(prog="colab_llm_hub.hyperqwen_launch")
    p.add_argument("--root", required=True, help="HyperQwen checkout with venv/ (install.ensure_hyperqwen)")
    p.add_argument("--model", required=True, help="downloaded checkpoint directory")
    p.add_argument("--port", type=int, default=30030)
    p.add_argument("--ctx", type=int, default=SHORT_CTX)
    p.add_argument("--mode", choices=("single", "batch"), default="single")
    p.add_argument("--spec", choices=("mtp", "dflash2", "off"), default="mtp")
    p.add_argument("--thinking", choices=("on", "off"), default="off")
    p.add_argument("--work-dir", default=None, help="local disk for the prepared copy (default: beside --model)")
    a = p.parse_args(argv)
    root = Path(a.root)
    python = str(root / "venv" / "bin" / "python")
    if not Path(python).is_file():
        raise SystemExit(f"[hyperqwen] {python} not found; run install.ensure_hyperqwen first")
    model = prepare(root, a.model, python, work_dir=a.work_dir)
    # The drafter is a separate checkpoint; it stays in the download, not in the prepared copy.
    env = launch_env(model, a.port, a.ctx, a.mode, a.spec, drafter=Path(a.model) / "dflash2",
                     thinking=a.thinking == "on")
    script = script_for(root, a.mode)
    print(f"[hyperqwen] exec bash {script} (mode {a.mode}, spec {a.spec}, ctx {a.ctx})", flush=True)
    os.chdir(root)
    os.execvpe("bash", ["bash", script], env)


if __name__ == "__main__":
    sys.exit(main())
