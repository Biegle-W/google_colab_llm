"""Model files live on Google Drive; the Colab VM's local disk holds only what streams from SSD.

Models are downloaded from Hugging Face straight to Drive once. A model that fits in VRAM + RAM is then
read directly from the Drive mount (resolve()); a model that streams part of its weights from disk during
inference gets those files copied to the runtime's local disk every run, with the rest linked from Drive.
A marker file written last makes copies atomic: a half-copied model is never mistaken for a complete one.
"""
import fcntl
import fnmatch
import json
import os
import shutil
import threading
import time
from pathlib import Path

MARKER = ".complete.json"
# Engines that memory-map their weights and read them in small scattered pieces, which the Drive mount serves at
# a few MB/s: their models are read once, start to finish, before the engine starts (see warm_read).
WARM_READ_BACKENDS = ("strata", "maya", "sparsa", "basalt", "strata_nvfp4")
USED = ".last_used"
CHUNK = 16 * 1024 * 1024


class StoreError(RuntimeError):
    pass


def hf_download(repo, patterns, dest, token=None, exclude=None):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, allow_patterns=patterns, ignore_patterns=exclude or None,
                      local_dir=str(dest), token=token)


def hf_list_files(repo, token=None):
    """[(path, size_bytes)] for every file in a Hugging Face model repo."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True, token=token)
    return [(s.rfilename, s.size) for s in info.siblings]


def hf_fetch_file(repo, path, dest, token=None):
    """Download one file into dest (keeping its repo path) and return the local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=path, local_dir=str(dest), token=token)


def select_files(files, allow=None, exclude=None):
    """Filter [(path, size)] the way snapshot_download's allow_patterns / ignore_patterns do."""
    def norm(patterns):
        return [p + "*" if p.endswith("/") else p for p in patterns] if patterns else None
    allow, exclude = norm(allow), norm(exclude)
    return [(p, n) for p, n in files
            if (allow is None or any(fnmatch.fnmatchcase(p, a) for a in allow))
            and not (exclude and any(fnmatch.fnmatchcase(p, x) for x in exclude))]


def _download_kwargs(entry):
    """Optional catalog field `exclude`: patterns to skip (duplicate weight variants, legacy checkpoints)."""
    return {"exclude": list(entry["exclude"])} if entry.get("exclude") else {}


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


def model_disk_gb(local_dir):
    """GB a model can use on the local disk: free space plus complete cached models, which the store evicts
    (they stay on Drive) when it needs room."""
    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)
    cached = sum(p.stat().st_size for d in root.iterdir() if d.is_dir() and is_complete(d)
                 for p in d.rglob("*") if p.is_file())
    return round((shutil.disk_usage(root).free + cached) / 1e9, 1)


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

    def model_path(self, entry, base=None):
        """What the engine is pointed at: a GGUF file for llama.cpp, a model directory otherwise."""
        base = Path(base) if base is not None else self.local_path(entry)
        if entry["backend"] in ("llama", "ninfer", "basalt"):
            return base / entry["model_file"]
        # Strata / Maya / vLLM / SGLang / Diffusers all take a directory of shards.
        return base

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
            # Strata/Maya DeepSeek packs often download through the engine itself (signed feeds /
            # setup.sh), not a plain HF snapshot. Skip HF when flagged; the engine uses data-dir.
            if entry.get("engine_download") and not is_complete(drive) and not is_complete(local):
                local.mkdir(parents=True, exist_ok=True)
                self._set(entry, "engine_managed")
                self.touch(entry)
                self.log(f"[store] {entry['id']}: files come from {entry['backend']} on first start "
                         f"(engine_download); Drive path {drive}")
                return self.model_path(entry)
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
            self.downloader(entry["repo"], entry["files"], local, self.token, **_download_kwargs(entry))
            # Optional companion packs (e.g. TensorFold DFlash2 drafter) land in a subdirectory.
            for extra in entry.get("extra_downloads") or []:
                dest = local / extra.get("subdir", "extra")
                dest.mkdir(parents=True, exist_ok=True)
                self.log(f"[store] {entry['id']}: also downloading {extra['repo']} → {dest.name}/")
                self.downloader(extra["repo"], extra.get("files") or ["*"], dest, self.token)
        finally:
            stop.set()
        write_marker(local, entry["repo"])
        if entry.get("model_file") and not (local / entry["model_file"]).is_file():
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

    def download_to_drive(self, entry, staging=None, lister=hf_list_files, fetch=hf_fetch_file, copier=None):
        """Download a model straight to Drive one file at a time; local disk only holds two files at once.

        For preparing models on a small CPU runtime. Files already on Drive with the right size are skipped,
        so an interrupted run (or Drive's daily upload limit) resumes where it stopped."""
        drive = self.drive_path(entry)
        if is_complete(drive):
            self.log(f"[store] {entry['id']}: already on Drive")
            return drive
        if entry.get("engine_download"):
            raise StoreError(f"{entry['id']} is downloaded by {entry['backend']} itself on first start; "
                             "prepare it on the GPU runtime")
        if self.downloader is not hf_download and lister is hf_list_files:
            # A custom whole-repo downloader (tests, mirrors): fetch locally, save to Drive, drop the local copy.
            self._download(entry, self.local_path(entry))
            self.save_to_drive(entry)
            shutil.rmtree(self.local_path(entry), ignore_errors=True)
            return drive
        sources = [(entry["repo"], entry["files"], entry.get("exclude"), "")]
        sources += [(x["repo"], x.get("files") or ["*"], x.get("exclude"), x.get("subdir", "extra"))
                    for x in entry.get("extra_downloads") or []]
        jobs = []
        for repo, allow, exclude, sub in sources:
            files = select_files(lister(repo, self.token), allow, exclude)
            if not files:
                raise StoreError(f"no files in {repo} match {allow}")
            jobs += [(repo, path, size, f"{sub}/{path}" if sub else path) for path, size in files]
        total = sum(size or 0 for *_, size, _ in jobs)
        self.progress[entry["id"]] = {"phase": "downloading to Drive", "done_bytes": 0, "total_bytes": total}
        self.log(f"[store] {entry['id']}: {len(jobs)} files, {total / 1e9:.1f} GB, straight to {drive}")
        if (drive / MARKER).exists():
            (drive / MARKER).unlink()                     # incomplete: rebuild the file list at the end
        staging = Path(staging or self.local_root / ".staging" / entry["id"])
        staging.mkdir(parents=True, exist_ok=True)
        # At most two files are local at once (one copying to Drive, the next downloading). Files not yet on
        # Drive need that room, less whatever an interrupted run already left in staging (it is reused).
        pending = sorted(((size or 0) for _, path, size, rel in jobs
                          if not ((drive / rel).is_file() and size is not None and (drive / rel).stat().st_size == size)),
                         reverse=True)
        staged = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
        free_bytes = shutil.disk_usage(staging).free
        # Up to three copies of a file size are local at once: the next file downloading, the staged one being
        # copied, and the Drive client's cache of that copy (two with a single file). A larger file is
        # downloaded straight into the Drive folder instead (measured: 2 x 80.6 GB staged + the cache filled 225 GB).
        direct_over = max(0, free_bytes - 20e9) / (3 if len(pending) > 1 else 2)   # one file: no overlap
        local_pending = [n for n in pending if n <= direct_over]
        need = max(0, sum(local_pending[:2]) - staged) / 1e9 + 1
        if free_bytes / 1e9 < need:
            raise StoreError(f"{entry['id']}: local disk needs {need:.0f} GB free for the next files")
        copier = copier or copy_file_atomic
        pending = None                                    # (thread, local file): copy overlaps the next download

        def finish(job):
            if job:
                job[0].join()
                if job[2]:
                    raise job[2][0]
        try:
            for repo, path, size, rel in jobs:
                dst = drive / rel
                if dst.is_file() and size is not None and dst.stat().st_size == size:
                    self._add(entry, size)
                    continue
                if size and size > direct_over:
                    finish(pending)
                    pending = None
                    self._fetch_direct(entry, repo, path, size, drive, rel, fetch, copier)
                    continue
                local = Path(fetch(repo, path, staging, self.token))
                finish(pending)
                errors = []

                def copy(local=local, dst=dst, errors=errors):
                    try:
                        copier(local, dst, lambda n: self._add(entry, n))
                        os.remove(local)
                    except Exception as e:                # surfaced by finish()
                        errors.append(e)
                t = threading.Thread(target=copy, daemon=True)
                t.start()
                pending = (t, local, errors)
            finish(pending)
        finally:
            if pending:
                pending[0].join()
        shutil.rmtree(staging, ignore_errors=True)
        # Mark complete only when every file on Drive has the size the source lists: a copy cut off midway
        # (a killed process, a stuck Drive client) must never be recorded as a finished model.
        expected = {rel: size for _, _, size, rel in jobs}
        wrong = [f"{rel} ({(drive / rel).stat().st_size if (drive / rel).is_file() else 'missing'} != {n})"
                 for rel, n in expected.items()
                 if n is not None and not ((drive / rel).is_file() and (drive / rel).stat().st_size == n)]
        if wrong:
            raise StoreError(f"{entry['id']}: files on Drive do not match the source: {', '.join(wrong[:3])}")
        files = {rel: (n if n is not None else (drive / rel).stat().st_size) for rel, n in expected.items()}
        (drive / MARKER).write_text(json.dumps({"repo": entry["repo"], "files": files}))
        if entry.get("model_file") and not (drive / entry["model_file"]).is_file():
            raise StoreError(f"{entry['model_file']} was not found in {entry['repo']}")
        self.progress[entry["id"]]["phase"] = "on Drive"
        self.log(f"[store] {entry['id']}: saved to Drive")
        return drive

    def _fetch_direct(self, entry, repo, path, size, drive, rel, fetch, copier):
        """Download one large file into the model's Drive folder, then wait for its upload (paced copiers)."""
        tx = getattr(copier, "tx", None)
        start = tx() if tx else 0
        self.log(f"[store] {entry['id']}: {rel} ({size / 1e9:.1f} GB) is larger than the local staging room; "
                 "downloading it straight into the Drive folder")
        base = drive / (rel[: -len(path)] if rel != path and rel.endswith(path) else "")
        got = Path(fetch(repo, path, base, self.token))
        if got.resolve() != (drive / rel).resolve():
            (drive / rel).parent.mkdir(parents=True, exist_ok=True)
            os.replace(got, drive / rel)
        shutil.rmtree(base / ".cache", ignore_errors=True)     # hf_hub_download's local_dir metadata
        self._add(entry, size)
        wait = getattr(copier, "wait_uploaded", None)
        if wait:
            wait(size, start, os.path.basename(rel))

    def resolve(self, entry, protect=()):
        """Where the engine reads the model this session.

        VRAM + RAM models: straight from Drive (downloaded there first if missing), nothing local.
        SSD-streamed models: their streamed files (`stream_files`, default all) copied to local disk every
        run, the rest symlinked from Drive, so the engine still sees one directory."""
        from . import catalog
        if entry.get("engine_download"):
            return self.ensure_local(entry, protect=protect)     # the engine fetches these itself
        drive = self.drive_path(entry)
        if not is_complete(drive):
            if is_complete(self.local_path(entry)):              # downloaded locally by an older version
                self.save_to_drive(entry)
            else:
                self.download_to_drive(entry, staging=self.local_root / ".staging" / entry["id"])
        if not catalog.streams_from_disk(entry):
            if entry.get("warm_read", entry["backend"] in WARM_READ_BACKENDS):
                self.warm_read(entry, drive)
            self._set(entry, "reading from Drive", entry["size_gb"] * 1e9)
            return self.model_path(entry, base=drive)
        return self._stage_streamed(entry, drive, protect)

    def warm_read(self, entry, drive, chunk=64 * 1024 * 1024, workers=4):
        """Read the model's files once, start to finish, before the engine starts.

        Engines that memory-map their weights (Basalt, the Strata NVFP4 fork) then read them in small scattered
        pieces, which the Drive mount serves at a few MB/s (measured: 9 of 68 GiB in 15 min). A sequential pass
        runs at the mount's full speed and leaves the data in the Drive client's local cache and the page cache,
        where the engine's own reads then find it. Files are read in parallel slices."""
        import concurrent.futures
        files = [p for p in Path(drive).rglob("*") if p.is_file() and p.name != MARKER]
        total = sum(p.stat().st_size for p in files)
        self.progress[entry["id"]] = {"phase": "reading from Drive (warming the cache)", "done_bytes": 0,
                                      "total_bytes": total}
        jobs = []
        for p in files:
            size = p.stat().st_size
            step = max(chunk, -(-size // workers))
            jobs += [(p, off, min(step, size - off)) for off in range(0, size, step)]
        t0 = time.time()

        def read(job):
            path, off, length = job
            with open(path, "rb", buffering=0) as f:
                f.seek(off)
                left = length
                while left > 0:
                    got = f.read(min(chunk, left))
                    if not got:
                        break
                    left -= len(got)
                    self._add(entry, len(got))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(read, jobs))
        dt = max(time.time() - t0, 1e-6)
        self.log(f"[store] {entry['id']}: read {total / 1e9:.1f} GB from Drive in {dt / 60:.1f} min "
                 f"({total / dt / 1e6:.0f} MB/s) before starting the engine")

    def _stage_streamed(self, entry, drive, protect):
        lock = self._lock(entry)
        try:
            local = self.local_path(entry)
            if is_complete(local):
                self.touch(entry)
                return self.model_path(entry)
            marker = json.loads((drive / MARKER).read_text())
            patterns = entry.get("stream_files") or ["*"]
            streamed = {rel: n for rel, n in marker["files"].items()
                        if any(fnmatch.fnmatchcase(rel, p) for p in patterns)}
            need = sum(streamed.values()) / 1e9
            self.evict_for(dict(entry, size_gb=need), protect)
            self._set(entry, "copying streamed files to local disk")
            self.progress[entry["id"]]["total_bytes"] = int(need * 1e9)
            self.log(f"[store] {entry['id']}: copying {len(streamed)} streamed files ({need:.1f} GB) to {local}")
            for rel, n in marker["files"].items():
                dst = local / rel
                if rel in streamed:
                    if not (dst.is_file() and not dst.is_symlink() and dst.stat().st_size == n):
                        copy_file_atomic(drive / rel, dst, lambda b: self._add(entry, b))
                    else:
                        self._add(entry, n)
                elif not dst.is_symlink():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        dst.unlink()
                    dst.symlink_to(drive / rel)                 # read from Drive like any other model
            shutil.copyfile(drive / MARKER, local / MARKER)     # marker last
            self._set(entry, "ready", entry["size_gb"] * 1e9)
            self.touch(entry)
            return self.model_path(entry)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def verify_against_source(self, entry, lister=hf_list_files):
        """Problems with a model marked complete on Drive, checked against the source's file list and sizes
        (empty = good). Catches files truncated before the marker was written."""
        drive = self.drive_path(entry)
        problems = []
        sources = [(entry["repo"], entry["files"], entry.get("exclude"), "")]
        sources += [(x["repo"], x.get("files") or ["*"], x.get("exclude"), x.get("subdir", "extra"))
                    for x in entry.get("extra_downloads") or []]
        for repo, allow, exclude, sub in sources:
            for path, size in select_files(lister(repo, self.token), allow, exclude):
                p = drive / (f"{sub}/{path}" if sub else path)
                if not p.is_file():
                    problems.append(f"{p.name} missing")
                elif size is not None and p.stat().st_size != size:
                    problems.append(f"{p.name} is {p.stat().st_size} bytes, source has {size}")
        return problems

    def prefetch(self, entry, protect=()):
        """Make sure the model is saved on Drive (downloading it if needed) without keeping the hub waiting on it."""
        if is_complete(self.drive_path(entry)) or entry.get("engine_download"):
            return
        self.download_to_drive(entry, staging=self.local_root / ".staging" / entry["id"])
