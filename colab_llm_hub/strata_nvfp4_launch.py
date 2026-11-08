"""Start the Strata NVFP4 fork (github.com/sergqwer/strata-nvfp4) for the hub.

The fork's ./setup.sh compiles the engine on Linux, checks the model files and writes the server config
(strata-<family>.json); serve/server.py then serves it. Setup expects its models at
<models>/<owner>--<repo>/<repo path>, each file checked against a pinned SHA-256 whose result it keeps in a
<file>.done mark. Here the files are the hub's copy on Google Drive, so:

1. <models>/<owner>--<repo> is a symlink to the hub's Drive folder for the model;
2. a .done mark is written beside each file whose size matches the size setup pins, with the SHA-256 setup
   pins, once the hub has recorded the folder complete (its sizes match Hugging Face's). Hashing 129 GB
   over the Drive mount at every start would take hours; the marks stay on Drive;
3. setup runs once (--yes --no-start); the built folder is then cached on Drive, and later sessions restore
   it and go straight to the server.

    python -m colab_llm_hub.strata_nvfp4_launch --root /content/strata-nvfp4 --model <drive dir> \\
        --family qwen-nvfp4-gptq --port 30055 --ctx 131072 --builds <Drive builds dir>
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

MARKER = ".hub_setup.json"      # in root: the settings the last successful setup ran with


def pinned_files(root, python, family):
    """{repo path: (bytes, sha256)} and the repo name setup pins for `family`, read from the fork's setup.py."""
    code = ("import json, setup; fam = setup.NVFP4_FAMILIES[%r]; repo = fam['sources']['pack']; "
            "t = setup.NVFP4_REPOS[repo]; "
            "print(json.dumps({'repo': repo, 'files': {p: v for c in setup.NVFP4_COMPONENTS "
            "for p, v in (t.get(c) or {}).items()}}))" % family)
    out = subprocess.run([python, "-c", code], cwd=str(root), capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"[strata-nvfp4] could not read the pinned files from setup.py:\n{out.stderr[-2000:]}")
    data = json.loads(out.stdout.strip().splitlines()[-1])
    return data["repo"], {p: tuple(v) for p, v in data["files"].items()}


def link_model(models_dir, repo, src):
    """<models>/<owner>--<repo> -> the hub's folder (replaced if it points elsewhere)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    link = models_dir / repo.replace("/", "--")
    if link.is_symlink() and os.readlink(link) == str(src):
        return link
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        raise SystemExit(f"[strata-nvfp4] {link} is a real folder; move it away so the Drive copy is used")
    link.symlink_to(src, target_is_directory=True)
    return link


def write_marks(src, pins, log=print):
    """The .done marks setup reads, for files the hub verified complete and whose size matches setup's pin."""
    src = Path(src)
    if not (src / ".complete.json").is_file():
        raise SystemExit(f"[strata-nvfp4] {src} is not a complete hub copy (.complete.json missing)")
    missing, wrote = [], 0
    for path, (size, sha) in pins.items():
        f = src / path
        if not f.is_file() or f.stat().st_size != size:
            missing.append(f"{path} ({f.stat().st_size if f.is_file() else 'missing'} != {size})")
            continue
        mark = f.with_name(f.name + ".done")
        try:
            if f"sha256 {sha}" in mark.read_text(encoding="utf-8", errors="replace"):
                continue
        except OSError:
            pass
        mark.write_text(f"sha256 {sha}\n", encoding="utf-8")
        wrote += 1
    if missing:
        raise SystemExit("[strata-nvfp4] the Drive copy does not match the revision setup pins: " +
                         ", ".join(missing[:4]) + " (save the model again from the prepare notebook)")
    if wrote:
        log(f"[strata-nvfp4] marked {wrote} files as checked (sizes match setup's pins)")


def setup_args(family, models_dir, data_dir, port, ctx, vision="no"):
    return ["--family", family, "--yes", "--no-start", "--no-browser", "--models-dir", str(models_dir),
            "--data-dir", str(data_dir), "--port", str(port), "--context", str(int(ctx)), "--vision", vision,
            "--host", "127.0.0.1"]


def find_config(root, family=None):
    """The strata-<family>.json setup wrote last."""
    cands = sorted(glob.glob(os.path.join(str(root), "strata-*.json")), key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None


def server_args(python, config, port):
    return [python, "serve/server.py", "--engine", "strata", "--config", str(config), "--port", str(port),
            "--host", "127.0.0.1"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="colab_llm_hub.strata_nvfp4_launch")
    p.add_argument("--root", required=True, help="the fork's checkout with .venv (install.ensure_strata_nvfp4)")
    p.add_argument("--model", required=True, help="the hub's folder of the model (on Drive)")
    p.add_argument("--family", default="qwen-nvfp4-gptq")
    p.add_argument("--port", type=int, default=30055)
    p.add_argument("--ctx", type=int, default=131072)
    p.add_argument("--builds", default=None, help="Drive builds folder: cache the built checkout there")
    p.add_argument("--tarball", default=None)
    a = p.parse_args(argv)
    root = Path(a.root)
    python = str(root / ".venv" / "bin" / "python")
    if not Path(python).is_file():
        raise SystemExit(f"[strata-nvfp4] {python} not found; run install.ensure_strata_nvfp4 first")
    models_dir, data_dir = root / "hub-models", root / "hub-data"
    repo, pins = pinned_files(root, python, a.family)
    link_model(models_dir, repo, Path(a.model).resolve())
    write_marks(a.model, pins)
    want = {"family": a.family, "ctx": a.ctx, "port": a.port, "model": str(a.model)}
    marker = root / MARKER
    have = json.loads(marker.read_text()) if marker.is_file() else None
    config = find_config(root, a.family)
    if have != want or not config:
        cmd = ["sh", "./setup.sh"] + setup_args(a.family, models_dir, data_dir, a.port, a.ctx)
        print("[strata-nvfp4] " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(root), env=dict(os.environ, STRATA_UPDATE_CHECK="0"))
        config = find_config(root, a.family)
        if r.returncode or not config:
            raise SystemExit(f"[strata-nvfp4] setup failed (exit {r.returncode}); see the log above")
        marker.write_text(json.dumps(want))
        if a.builds and a.tarball:
            tmp = a.tarball + ".tmp"
            print(f"[strata-nvfp4] caching the built folder on Drive: {a.tarball}", flush=True)
            r = subprocess.run(["tar", "-cf", tmp, "-C", str(root.parent), "--exclude", f"{root.name}/hub-models",
                                "--exclude", f"{root.name}/hub-data", root.name])
            if r.returncode == 0:
                os.replace(tmp, a.tarball)
    args = server_args(python, config, a.port)
    print(f"[strata-nvfp4] exec {' '.join(args)}", flush=True)
    os.chdir(root)
    os.execvpe(args[0], args, dict(os.environ, STRATA_UPDATE_CHECK="0"))


if __name__ == "__main__":
    sys.exit(main())
