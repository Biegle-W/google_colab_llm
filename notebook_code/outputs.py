"""Keep every generated image and video on Google Drive, whoever asked for it.

The image servers (ComfyUI, Diffusers) call save() for each output they return, so work made from a harness
over the API is kept too, not only on the machine that asked. Files go to <Drive>/outputs/images/ (the folder
the notebooks' own test cells use), each with a .json beside it holding the prompt and settings.
"""
import json
import os
import threading
import time
from pathlib import Path

DEFAULT_DRIVE = "/content/drive/MyDrive/colab_model_server"


def folder(drive_dir=None):
    """<Drive>/outputs/images, or None when Drive is not mounted (the local no-Drive fallback keeps nothing)."""
    drive = drive_dir or os.environ.get("COLAB_LLM_DRIVE") or DEFAULT_DRIVE
    if os.environ.get("COLAB_LLM_NO_DRIVE") == "1" or not os.path.isdir(drive):
        return None
    return Path(drive) / "outputs" / "images"


def _name(model_id, ext, index, stamp):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (model_id or "image"))[:48]
    return f"{stamp}-{safe}" + (f"-{index + 1}" if index else "") + f".{ext}"


def plan(model_id, count, ext="png", drive_dir=None, stamp=None):
    """The Drive paths save() will write for `count` outputs (returned to the caller before the slow write)."""
    base = folder(drive_dir)
    if base is None:
        return []
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    return [base / _name(model_id, ext, i, stamp) for i in range(count)]


def save(blobs, model_id, prompt="", ext="png", extra=None, drive_dir=None, background=True, log=print):
    """Write each output (bytes) and a .json with its prompt to Drive. Returns the paths (as strings), or [] when
    Drive is not there. With `background` the writes happen in a thread, so a slow Drive never delays a reply."""
    paths = plan(model_id, len(blobs), ext, drive_dir)
    if not paths:
        return []
    meta = {"model": model_id, "prompt": prompt, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "api", **(extra or {})}

    def write():
        try:
            paths[0].parent.mkdir(parents=True, exist_ok=True)
            for path, blob in zip(paths, blobs):
                path.write_bytes(blob)
                path.with_suffix(path.suffix + ".json").write_text(
                    json.dumps({**meta, "file": path.name}, indent=2) + "\n")
        except OSError as e:                   # never fails a generation: the caller still has its output
            log(f"[outputs] could not save to Drive: {e}")
    if background:
        threading.Thread(target=write, daemon=True).start()
    else:
        write()
    return [str(p) for p in paths]


def sync_comfy(output_dir, drive_dir=None, done=None):
    """Copy what ComfyUI's own editor made (its output folder) to <Drive>/outputs/comfyui/, keeping subfolders.
    Files the API wrote (colab-* names) are skipped: save() already kept those. Returns how many were copied."""
    import shutil
    base = folder(drive_dir)
    src = Path(output_dir)
    if base is None or not src.is_dir():
        return 0
    dest = base.parent / "comfyui"
    done = done if done is not None else set()
    copied = 0
    for path in src.rglob("*"):
        if not path.is_file() or path.name.startswith("colab-") or path.suffix in (".tmp", ".part"):
            continue
        rel = path.relative_to(src)
        key = (str(rel), path.stat().st_size)
        if key in done:
            continue
        target = dest / rel
        try:
            if not (target.is_file() and target.stat().st_size == path.stat().st_size):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied += 1
            done.add(key)
        except OSError:
            pass                                  # Drive busy: tried again next round
    return copied


def watch_comfy(output_dir, drive_dir=None, interval=60, sleep=time.sleep):
    """Copy the editor's new outputs to Drive every `interval` seconds, in a background thread."""
    done = set()

    def loop():
        while True:
            sync_comfy(output_dir, drive_dir, done)
            sleep(interval)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread
