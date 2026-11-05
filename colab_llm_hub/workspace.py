"""Google Drive workspace layout for Colab: engines, config, logs, and outputs survive restarts.

Everything durable lives under ``drive_dir`` (default ``MyDrive/llm_models``). Local ``/content``
paths are only scratch space for engines that need fast disk; they are restored from Drive.
"""
import json
import os
import shutil
import time
from pathlib import Path

DEFAULT_DRIVE = "/content/drive/MyDrive/llm_models"
DEFAULT_LOCAL_MODELS = "/content/models"
SETTINGS_NAME = "settings.json"
DEFAULT_SETTINGS = {
    "prefetch": [],
    "engines": ["llama", "diffusers"],
    "ctx_override": {},
    "port": 8000,
    "tunnel": "quick",
    "llama_cpp_ref": "4c3c5810f87efe177949f1b85ae509d38a2a311f",
    "rebuild_llama": False,
    "refresh_wheels": False,
    "hub_ref": None,
}


class Workspace:
    def __init__(self, drive_dir=None, local_models=None):
        self.drive = Path(drive_dir or os.environ.get("COLAB_LLM_DRIVE", DEFAULT_DRIVE))
        self.local_models = Path(local_models or DEFAULT_LOCAL_MODELS)
        self.models = self.drive / "models"
        self.builds = self.drive / "builds"
        self.config = self.drive / "config"
        self.hub = self.drive / "hub"
        self.logs = self.drive / "logs"
        self.engine_logs = self.logs / "hub_logs"
        self.outputs = self.drive / "outputs"
        self.images = self.outputs / "images"
        self.chat = self.outputs / "chat"
        self.status = self.outputs / "status"
        self.hf_cache = self.drive / "hf_cache"
        self.tools = self.builds / "tools"
        self.hub_pkg = self.builds / "hub_pkg"
        self.settings_path = self.config / SETTINGS_NAME

    def ensure(self):
        for p in (self.models, self.builds, self.config, self.hub, self.logs, self.engine_logs,
                  self.outputs, self.images, self.chat, self.status, self.hf_cache, self.tools,
                  self.hub_pkg, self.local_models):
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
    def cloudflared_path(self):
        return self.tools / "cloudflared"

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


def open_workspace(drive_dir=None, local_models=None):
    return Workspace(drive_dir, local_models).ensure().apply_env()
