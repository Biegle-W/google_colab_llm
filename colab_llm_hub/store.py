"""Model files: a permanent copy on Google Drive, a fast working copy on the Colab VM's local disk.

Models are downloaded from Hugging Face once, saved to Drive, and copied back to local disk in later
sessions (engines load much faster from local disk than through the Drive mount). A marker file written
last makes copies atomic: a half-copied model is never mistaken for a complete one.
"""
import fcntl
import json
import os
import shutil
import threading
import time
from pathlib import Path

MARKER = ".complete.json"
USED = ".last_used"
CHUNK = 16 * 1024 * 1024


class StoreError(RuntimeError):
    pass


def hf_download(repo, patterns, dest, token=None):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, allow_patterns=patterns, local_dir=str(dest), token=token)


def _files(root):
    out = {}
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not rel.parts[0].startswith(".") and not p.name.endswith(".part"):
            out[rel.as_posix()] = p.stat().st_size
    return out


def is_complete(root):
    root = Path(root)
    try:
        listed = json.loads((root / MARKER).read_text())["files"]
    except (OSError, ValueError, KeyError):
        return False
    return bool(listed) and all((root / rel).is_file() and (root / rel).stat().st_size == size
                                for rel, size in listed.items())


def write_marker(root, repo):
    files = _files(root)
    if not files:
        raise StoreError(f"nothing was downloaded into {root}")
    (Path(root) / MARKER).write_text(json.dumps({"repo": repo, "files": files}))


def copy_file_atomic(src, dst, on_bytes=None):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    with open(src, "rb") as a, open(part, "wb") as b:
        while chunk := a.read(CHUNK):
            b.write(chunk)
            if on_bytes:
                on_bytes(len(chunk))
    os.replace(part, dst)


class ModelStore:
    def __init__(self, drive_dir, local_dir, token=None, downloader=hf_download, min_free_gb=8, log=print):
        self.drive_root = Path(drive_dir) / "models"
        self.local_root = Path(local_dir)
        self.token, self.downloader, self.min_free_gb, self.log = token, downloader, min_free_gb, log
        self.drive_root.mkdir(parents=True, exist_ok=True)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.progress = {}          # model id -> {"phase", "done_bytes", "total_bytes"}
        self._threads = []
        self._copying = set()       # local dirs being copied to Drive: never evict these

    # ---- locations -------------------------------------------------------------------------
    def drive_path(self, entry): return self.drive_root / entry["id"]
    def local_path(self, entry): return self.local_root / entry["id"]

    def model_path(self, entry):
        """What the engine is pointed at: a GGUF file for llama.cpp, a model directory for vLLM."""
        base = self.local_path(entry)
        return base / entry["model_file"] if entry["backend"] == "llama" else base

    def state(self, entry):
        return {"on_drive": is_complete(self.drive_path(entry)), "local": is_complete(self.local_path(entry)),
                **self.progress.get(entry["id"], {})}

    # ---- progress --------------------------------------------------------------------------
    def _set(self, entry, phase, done=0):
        self.progress[entry["id"]] = {"phase": phase, "done_bytes": int(done),
                                      "total_bytes": int(entry["size_gb"] * 1e9)}

    def _add(self, entry, nbytes):
        p = self.progress[entry["id"]]
        p["done_bytes"] += nbytes

    # ---- disk space ------------------------------------------------------------------------
    def local_models(self):
        return [d for d in self.local_root.iterdir() if d.is_dir() and is_complete(d)]

    def free_gb(self):
        return shutil.disk_usage(self.local_root).free / 1e9

    def evict_for(self, entry, protect=()):
        """Delete least-recently-used local copies (they stay on Drive) until the model fits."""
        need = entry["size_gb"] + self.min_free_gb
        protected = ({str(self.local_path(e)) for e in protect} | {str(self.local_path(entry))}
                     | {str(self.local_root / i) for i in self._copying})
        while self.free_gb() < need:
            victims = sorted((d for d in self.local_models() if str(d) not in protected),
                             key=lambda d: (d / USED).stat().st_mtime if (d / USED).exists() else 0)
            if not victims:
                raise StoreError(f"Not enough local disk for {entry['id']}: need {need:.0f} GB free, "
                                 f"have {self.free_gb():.0f} GB and nothing left to evict.")
            self.log(f"[store] freeing local disk: removing {victims[0].name} (kept on Drive if saved)")
            shutil.rmtree(victims[0])

    def touch(self, entry):
        p = self.local_path(entry) / USED
        if p.parent.exists():
            p.write_text(str(time.time()))

    # ---- main entry points -----------------------------------------------------------------
    def _lock(self, entry):
        self.local_root.mkdir(parents=True, exist_ok=True)      # /content is wiped when the runtime restarts
        f = open(self.local_root / f".{entry['id']}.lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def ensure_local(self, entry, protect=(), background_drive_copy=True):
        """Make the model available on local disk and return the path to give the engine."""
        lock = self._lock(entry)
        try:
            local, drive = self.local_path(entry), self.drive_path(entry)
            if is_complete(local):
                self.touch(entry)
                return self.model_path(entry)
            if (local / MARKER).exists():
                shutil.rmtree(local)               # marked complete once, but files are missing or changed
            # Otherwise keep partial downloads: Hugging Face resumes them.
            self.evict_for(entry, protect)
            if is_complete(drive):
                self._copy_tree(entry, drive, local, "copying from Drive")
            else:
                self._download(entry, local)
                if background_drive_copy:
                    self._drive_copy_async(entry)
            self.touch(entry)
            return self.model_path(entry)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _copy_tree(self, entry, src, dst, phase):
        marker = json.loads((Path(src) / MARKER).read_text())
        self._set(entry, phase)
        self.log(f"[store] {entry['id']}: {phase} ({entry['size_gb']:.1f} GB)")
        for rel in marker["files"]:
            copy_file_atomic(Path(src) / rel, Path(dst) / rel, lambda n: self._add(entry, n))
        shutil.copyfile(Path(src) / MARKER, Path(dst) / MARKER)       # marker last
        self._set(entry, "ready", entry["size_gb"] * 1e9)

    def _download(self, entry, local):
        self._set(entry, "downloading")
        self.log(f"[store] {entry['id']}: downloading {entry['repo']} ({entry['size_gb']:.1f} GB)")
        stop = threading.Event()

        def watch():                                  # snapshot_download has no byte callback; watch the directory
            while not stop.wait(3):
                try:
                    self.progress[entry["id"]]["done_bytes"] = sum(
                        p.stat().st_size for p in local.rglob("*") if p.is_file())
                except OSError:
                    pass
        t = threading.Thread(target=watch, daemon=True)
        t.start()
        try:
            local.mkdir(parents=True, exist_ok=True)
            self.downloader(entry["repo"], entry["files"], local, self.token)
        finally:
            stop.set()
        write_marker(local, entry["repo"])
        if entry["backend"] == "llama" and not (local / entry["model_file"]).is_file():
            raise StoreError(f"{entry['model_file']} was not found in {entry['repo']}")
        self._set(entry, "ready", entry["size_gb"] * 1e9)

    def save_to_drive(self, entry):
        """Copy a complete local model to Drive (no-op when Drive already has it)."""
        local, drive = self.local_path(entry), self.drive_path(entry)
        if is_complete(drive) or not is_complete(local):
            return
        if drive.exists():
            shutil.rmtree(drive)
        self._copy_tree(entry, local, drive, "saving to Drive")

    def _drive_copy_async(self, entry):
        self._copying.add(entry["id"])

        def run():
            try:
                self.save_to_drive(entry)
                self.log(f"[store] {entry['id']}: saved to Drive")
            except Exception as e:                      # keep serving from local disk; Drive copy retried next time
                self.log(f"[store] {entry['id']}: saving to Drive failed: {e}")
            finally:
                self._copying.discard(entry["id"])
        t = threading.Thread(target=run, name=f"drive-copy-{entry['id']}", daemon=True)
        t.start()
        self._threads.append(t)

    def wait_for_drive_copies(self):
        for t in self._threads:
            t.join()

    def prefetch(self, entry, protect=()):
        """Make sure the model is saved on Drive (downloading it if needed) without keeping the hub waiting on it."""
        if is_complete(self.drive_path(entry)):
            return
        self.ensure_local(entry, protect=protect, background_drive_copy=False)
        self.save_to_drive(entry)
