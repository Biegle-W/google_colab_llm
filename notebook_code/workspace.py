"""Google Drive workspace layout for Colab: engines, config, logs, and outputs survive restarts.

Everything durable lives under ``drive_dir`` (default ``MyDrive/colab_model_server``). Local ``/content``
paths are only scratch space for engines that need fast disk; they are restored from Drive.
"""
import json
import os
import shutil
import time
from pathlib import Path

DEFAULT_DRIVE = "/content/drive/MyDrive/colab_model_server"
# Folders under the Drive root (and what they were called before 2026-11-08).
MODELS, ENGINES, SETTINGS, ACCESS, LOGS, OUTPUTS, HUGGINGFACE = (
    "models", "engines", "settings", "access", "logs", "outputs", "huggingface")
ENGINE_LOGS = "engines"                   # under logs/
STUDIO_HUB = "unsloth_studio"             # under models/: Unsloth Studio's downloads (Hugging Face cache layout)
OLD_ROOT = "llm_models"
OLD_NAMES = {"builds": ENGINES, "config": SETTINGS, "hub": ACCESS, "hf_cache": HUGGINGFACE}
OLD_SUBDIRS = {(LOGS, "hub_logs"): ENGINE_LOGS, (MODELS, "hf_hub"): STUDIO_HUB}


def drive_path(drive_dir, folder, *parts):
    return os.path.join(str(drive_dir), folder, *parts)


def migrate_drive_layout(drive_dir=DEFAULT_DRIVE, log=print):
    """Rename the old Drive layout (MyDrive/llm_models with builds/ config/ hub/ hf_cache/) to the current one.

    Only renames (instant on Drive, nothing is copied), and only where the new name does not exist yet."""
    root = Path(drive_dir)
    old_root = root.parent / OLD_ROOT
    if root.name != OLD_ROOT and old_root.is_dir():
        if not root.exists():
            os.rename(old_root, root)
            log(f"Renamed {old_root} -> {root}")
        else:                       # another notebook already made the new root: move what it does not have
            for child in sorted(old_root.iterdir()):
                dst = root / OLD_NAMES.get(child.name, child.name)
                if not dst.exists():
                    os.rename(child, dst)
                    log(f"Moved {child} -> {dst}")
                elif child.is_dir() and dst.is_dir():
                    for sub in sorted(child.iterdir()):
                        if not (dst / sub.name).exists():
                            os.rename(sub, dst / sub.name)
            for d, _, _ in sorted(os.walk(old_root), key=lambda x: -len(x[0])):     # empty folders, deepest first
                try:
                    os.rmdir(d)
                except OSError:
                    pass
            if old_root.exists():
                log(f"{old_root} still has files that also exist in {root}; left in place")
    if not root.is_dir():
        return
    for old, new in list(OLD_NAMES.items()) + [(os.path.join(*k), os.path.join(k[0], v)) for k, v in OLD_SUBDIRS.items()]:
        src, dst = root / old, root / new
        if src.is_dir() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
            log(f"Renamed {src} -> {dst}")
DEFAULT_LOCAL_MODELS = "/content/models"
# Some runtimes (the A100 80 GB) add a fast local NVMe disk here; the G4 and L4 do not.
LOCAL_DISK_CANDIDATES = ("/mnt/local-scratch", "/content")


def pick_local_models(candidates=LOCAL_DISK_CANDIDATES, free_bytes=None):
    """Directory for the working copy of model files: the writable disk with the most free space.

    The working copy is a cache (models stay on Drive), so the largest fast local disk wins; on the A100 that
    is /mnt/local-scratch (~350 GB free) rather than /content (~250 GB). COLAB_LLM_LOCAL_MODELS overrides.
    """
    override = os.environ.get("COLAB_LLM_LOCAL_MODELS")
    if override:
        return Path(override)
    free_bytes = free_bytes or (lambda p: shutil.disk_usage(p).free)
    best, best_free = None, -1
    for root in candidates:
        if not (os.path.isdir(root) and os.access(root, os.W_OK)):
            continue
        try:
            free = free_bytes(root)
        except OSError:
            continue
        if free > best_free:
            best, best_free = root, free
    return Path(best) / "models" if best else Path(DEFAULT_LOCAL_MODELS)


def local_drive_dir():
    """Stand-in for the Drive folder when Google Drive is not available: the largest local disk. Models
    download straight to this runtime and are gone when it stops."""
    return pick_local_models().parent / "colab_model_server"


SETTINGS_NAME = "settings.json"
DEFAULT_SETTINGS = {
    "prefetch": [],
    "engines": ["llama", "diffusers"],
    "ctx_override": {},
    "port": 8000,
    "tunnel": "tailscale",
    "llama_cpp_ref": "4c3c5810f87efe177949f1b85ae509d38a2a311f",
    "rebuild_llama": False,
    "refresh_wheels": False,
    "hub_ref": None,
}


class Workspace:
    def __init__(self, drive_dir=None, local_models=None, settings_name=None, engine=None):
        self.drive = Path(drive_dir or os.environ.get("COLAB_LLM_DRIVE", DEFAULT_DRIVE))
        self.local_models = Path(local_models) if local_models else pick_local_models()
        self.engine = engine
        self.models = self.drive / MODELS
        self.builds = self.drive / ENGINES          # engine builds, wheels, tools
        self.config = self.drive / SETTINGS
        self.hub = self.drive / ACCESS              # API key, tunnel keys, endpoint, dsh provider
        self.logs = self.drive / LOGS
        self.engine_logs = self.logs / ENGINE_LOGS
        self.outputs = self.drive / OUTPUTS
        self.images = self.outputs / "images"
        self.chat = self.outputs / "chat"
        self.status = self.outputs / "status"
        self.hf_cache = self.drive / HUGGINGFACE
        self.tools = self.builds / "tools"
        if settings_name:
            name = settings_name
        elif engine:
            name = f"settings_{engine}.json"
        else:
            name = SETTINGS_NAME
        self.settings_path = self.config / name

    def ensure(self):
        for p in (self.models, self.builds, self.config, self.hub, self.logs, self.engine_logs,
                  self.outputs, self.images, self.chat, self.status, self.hf_cache, self.tools,
                  self.local_models):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def apply_env(self):
        """Point Hugging Face / Diffusers / Transformers caches at Drive."""
        os.environ["HF_HOME"] = str(self.hf_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.hf_cache / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(self.hf_cache / "transformers")
        os.environ["DIFFUSERS_CACHE"] = str(self.hf_cache / "diffusers")
        os.environ["COLAB_LLM_DRIVE"] = str(self.drive)
        return self

    # ---- settings -----------------------------------------------------------------
    def load_settings(self, defaults=None):
        base = dict(DEFAULT_SETTINGS)
        if defaults:
            base.update(defaults)
        if self.settings_path.is_file():
            try:
                saved = json.loads(self.settings_path.read_text())
                if isinstance(saved, dict):
                    base.update({k: saved[k] for k in saved if k in base or k in DEFAULT_SETTINGS
                                 or k in ("prefetch", "engines", "ctx_override", "notes")})
                    base.update(saved)
            except (OSError, ValueError):
                pass
        return base

    def save_settings(self, settings):
        self.config.mkdir(parents=True, exist_ok=True)
        data = dict(settings)
        # Never persist secrets into settings JSON (those live under hub/ on Drive only).
        for banned in ("tailscale_authkey", "ts_authkey", "authkey", "api_key",
                       "TAILSCALE_AUTHKEY", "CLOUDFLARE_TUNNEL_TOKEN"):
            data.pop(banned, None)
        data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = self.settings_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, self.settings_path)
        return self.settings_path

    # ---- outputs ------------------------------------------------------------------
    def save_json(self, folder, name, payload):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = folder / f"{stamp}-{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def save_image_b64(self, b64_json, model_id, prompt="", ext="png"):
        import base64
        self.images.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (model_id or "image"))[:48]
        path = self.images / f"{stamp}-{safe}.{ext}"
        path.write_bytes(base64.b64decode(b64_json))
        meta = {"model": model_id, "prompt": prompt, "file": path.name,
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
        return path

    def save_video_b64(self, b64_mp4, model_id, prompt=""):
        """A generated MP4 (with its audio track) next to the images, plus a .json with the prompt."""
        return self.save_image_b64(b64_mp4, model_id, prompt=prompt, ext="mp4")

    def save_chat(self, model_id, messages, response, usage=None):
        return self.save_json(self.chat, model_id or "chat", {
            "model": model_id, "messages": messages, "response": response, "usage": usage,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def save_endpoint(self, base_url, api_key_hint, gpu_name, models):
        payload = {"base_url": base_url, "api_key_file": str(self.hub / "api_key"),
                   "api_key_hint": (api_key_hint[:6] + "…") if api_key_hint else None,
                   "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": gpu_name,
                   "models": models}
        path = self.hub / "endpoint.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        self.save_json(self.status, "endpoint", payload)
        return path

    # ---- tool / package caches ----------------------------------------------------
    def mirror_log(self, local_path, name=None):
        """Copy a local log onto Drive (best-effort)."""
        local_path = Path(local_path)
        if not local_path.is_file():
            return None
        dest = self.logs / (name or local_path.name)
        try:
            shutil.copy2(local_path, dest)
            return dest
        except OSError:
            return None

    def describe(self):
        return {
            "drive": str(self.drive),
            "models": str(self.models),
            "builds": str(self.builds),
            "config": str(self.config),
            "logs": str(self.logs),
            "outputs": str(self.outputs),
            "hf_cache": str(self.hf_cache),
            "settings": str(self.settings_path),
            "settings_exists": self.settings_path.is_file(),
        }


def open_workspace(drive_dir=None, local_models=None, settings_name=None, engine=None):
    migrate_drive_layout(drive_dir or os.environ.get("COLAB_LLM_DRIVE", DEFAULT_DRIVE))
    return Workspace(drive_dir, local_models, settings_name=settings_name, engine=engine).ensure().apply_env()
