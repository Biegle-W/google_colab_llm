"""Command lines and process control for the inference engines.

llama.cpp flags follow the reference setup (flash attention, q8_0 KV cache, --jinja, no prompt cache in RAM,
thinking off by default). vLLM flags use the Qwen tool/reasoning parsers named in each catalog entry.
Diffusers runs as `python -m colab_llm_hub.diffusers_server` against a Drive-restored model directory.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORTS = {
    "llama": 30080, "vllm": 30010, "sglang": 30020, "diffusers": 30090,
    "strata": 30050, "maya": 30060, "tensorfold": 30070, "comfyui": 30095,
    "sparsa": 30065, "hyperqwen": 30030, "ninfer": 30040,
}
READY_TIMEOUT = 1800
CHAT_BACKENDS = ("llama", "vllm", "sglang", "strata", "maya", "tensorfold", "sparsa", "hyperqwen", "ninfer")
IMAGE_BACKENDS = ("diffusers", "comfyui")
# Engines that advertise their own model names; readiness = OpenAI /v1/models answers.
ALIAS_READY_BACKENDS = ("strata", "maya", "sparsa")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _context_flag(ctx):
    """Strata prefers 32k/64k/128k labels; Maya takes a token count."""
    n = int(ctx)
    if n >= 131072:
        return "128k"
    if n >= 65536:
        return "64k"
    if n >= 32768:
        return "32k"
    if n >= 16384:
        return "16k"
    if n >= 8192:
        return "8k"
    return str(n)


def llama_command(entry, model_path, ctx, llama_bin, port=None):
    args = [str(llama_bin), "-m", str(model_path), "--alias", entry["id"],
            "--host", "127.0.0.1", "--port", str(port or PORTS["llama"]),
            "-c", str(ctx), "-ngl", str(entry.get("ngl", 99)),
            "--jinja", "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--parallel", "1",
            "--batch-size", str(entry.get("batch", 2048)), "--ubatch-size", str(entry.get("ubatch", 512)),
            "--cache-ram", "0", "--ctx-checkpoints", "0"]
    if not entry.get("thinking", False):
        args += ["--reasoning", "off", "--chat-template-kwargs", json.dumps({"enable_thinking": False})]
    return args + list(entry.get("extra_args", []))


def vllm_command(entry, model_path, ctx, vllm_bin="vllm", port=None, gpu_util=0.90):
    args = [str(vllm_bin), "serve", str(model_path), "--served-model-name", entry["id"],
            "--host", "127.0.0.1", "--port", str(port or PORTS["vllm"]),
            "--max-model-len", str(ctx), "--gpu-memory-utilization", str(entry.get("gpu_util", gpu_util)),
            "--max-num-seqs", str(entry.get("max_num_seqs", 4))]
    if entry.get("tool_parser"):
        args += ["--enable-auto-tool-choice", "--tool-call-parser", entry["tool_parser"]]
    if entry.get("reasoning_parser"):
        args += ["--reasoning-parser", entry["reasoning_parser"]]
    if entry.get("language_model_only"):
        args += ["--language-model-only"]
    if not entry.get("thinking", False):
        args += ["--default-chat-template-kwargs", json.dumps({"enable_thinking": False})]
    return args + list(entry.get("extra_args", []))


def sglang_command(entry, model_path, ctx, sglang_bin=None, port=None):
    # `python -m sglang.launch_server` is the usual Colab install; allow an override binary.
    if sglang_bin:
        args = [str(sglang_bin)]
    else:
        args = [sys.executable, "-m", "sglang.launch_server"]
    args += ["--model-path", str(model_path), "--served-model-name", entry["id"],
             "--host", "127.0.0.1", "--port", str(port or PORTS["sglang"]),
             "--context-length", str(ctx),
             "--mem-fraction-static", str(entry.get("mem_fraction", 0.85))]
    if entry.get("tool_parser"):
        args += ["--tool-call-parser", entry["tool_parser"]]
    if entry.get("reasoning_parser"):
        args += ["--reasoning-parser", entry["reasoning_parser"]]
    if not entry.get("thinking", False):
        # SGLang's flag is --default-chat-template-kwargs (there is no --chat-template-kwargs).
        args += ["--default-chat-template-kwargs", json.dumps({"enable_thinking": False})]
    return args + list(entry.get("extra_args", []))


def diffusion_command(entry, model_path, max_edge, python=None, port=None):
    args = [python or sys.executable, "-m", "colab_llm_hub.diffusers_server",
            "--model-path", str(model_path), "--model-id", entry["id"],
            "--port", str(port or PORTS["diffusers"]), "--max-edge", str(max_edge),
            "--pipeline", entry.get("pipeline", "AutoPipelineForText2Image"),
            "--dtype", entry.get("dtype", "float16"),
            "--steps", str(entry.get("default_steps", 28)),
            "--guidance", str(entry.get("default_guidance", 7.5))]
    if entry.get("quantization"):
        args += ["--quantization", entry["quantization"]]
    return args + list(entry.get("extra_args", []))


def strata_command(entry, model_path, ctx, strata_bin="strata", port=None, data_dir=None):
    """Headless Strata Terminal launch (official: --yes --no-tui). OpenAI API on --port."""
    family = entry.get("strata_family", "qwen")
    model = entry.get("strata_model", "iq3_xxs")
    args = [str(strata_bin), "launch",
            "--family", str(family), "--model", str(model),
            "--context", _context_flag(ctx),
            "--yes", "--no-tui", "--restart",
            "--access", "local", "--no-api-key",
            "--port", str(port or PORTS["strata"]),
            "--vision", entry.get("vision", "off"),
            "--tune", "skip"]
    if data_dir:
        args += ["--data-dir", str(data_dir)]
    # Prefer files already restored from Drive; otherwise Strata downloads into --data-dir.
    if model_path and os.path.isdir(str(model_path)) and any(
            p.endswith(".gguf") for p in _walk_names(model_path)):
        args += ["--model-path", str(model_path)]
    return args + list(entry.get("extra_args", []))


def maya_command(entry, model_path, ctx, maya_root, port=None, models_dir=None):
    """Project Maya setup/start (official ./setup.sh). OpenAI API on --port."""
    setup = os.path.join(str(maya_root), "setup.sh")
    model = entry.get("maya_model", entry["id"])
    args = ["bash", setup, "--model", str(model),
            "--port", str(port or PORTS["maya"]),
            "--host", "127.0.0.1",
            "--yes", "--plain",
            "--drafting", entry.get("drafting", "off")]
    if models_dir:
        args += ["--models-dir", str(models_dir)]
    if model_path and os.path.isdir(str(model_path)) and any(
            p.endswith(".gguf") or p.endswith(".safetensors") for p in _walk_names(model_path)):
        args += ["--import-model", str(model_path)]
    else:
        args += ["--download-model"]
    # DeepSeek Maya quants also accept --ds4-quant when maya_model is the family name.
    if entry.get("ds4_quant"):
        args += ["--ds4-quant", str(entry["ds4_quant"])]
    if entry.get("mm3_quant"):
        args += ["--mm3-quant", str(entry["mm3_quant"])]
    return args + list(entry.get("extra_args", []))


def tensorfold_command(entry, model_path, ctx, tensorfold_bin, port=None):
    """Native TensorFold serve (official bin/tensorfold-native). OpenAI API on --port."""
    args = [str(tensorfold_bin), "serve", str(model_path),
            "--backend", "cuda", "--device", str(entry.get("device", 0)),
            "--host", "127.0.0.1", "--port", str(port or PORTS["tensorfold"]),
            "--name", entry["id"], "--context", str(int(ctx)),
            "--parallel", str(entry.get("parallel", 1)),
            "--temperature", "0", "--no-thinking", "--no-update-check",
            "--prompt-cache-gib", str(entry.get("prompt_cache_gib", 0))]
    drafter_subdir = entry.get("drafter_subdir", "drafter")
    drafter_path = os.path.join(str(model_path), drafter_subdir)
    if entry.get("drafter_bits") and os.path.isdir(drafter_path):
        args += ["--drafter", drafter_path, "--drafter-bits", str(entry["drafter_bits"])]
    return args + list(entry.get("extra_args", []))


def comfyui_command(entry, model_path, max_edge, comfy_root, python=None, port=None):
    """OpenAI-compatible ComfyUI wrapper (starts ComfyUI + /v1/images/generations)."""
    args = [python or sys.executable, "-m", "colab_llm_hub.comfyui_server",
            "--comfy-root", str(comfy_root),
            "--model-path", str(model_path), "--model-id", entry["id"],
            "--port", str(port or PORTS["comfyui"]),
            "--max-edge", str(max_edge),
            "--workflow", entry.get("workflow", "qwen-image-2.1-turbo.api.json"),
            "--steps", str(entry.get("default_steps", 8)),
            "--guidance", str(entry.get("default_guidance", 1.0))]
    if entry.get("kind") == "video":
        args += ["--kind", "video"]
    return args + list(entry.get("extra_args", []))


def sparsa_command(entry, model_path, ctx, sparsa_root, python=None, port=None):
    """Sparsa MiniMax-M3.1-Flash serve (official python sparsa.py serve)."""
    script = os.path.join(str(sparsa_root), "sparsa.py")
    args = [python or sys.executable, script, "serve",
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port or PORTS["sparsa"]),
            "--context", str(int(ctx)),
            "--no-api-key", "--no-vision"]
    engine = os.path.join(str(sparsa_root), "bin", "strata-minimax")
    if os.path.isfile(engine) and os.access(engine, os.X_OK):
        args += ["--engine", engine]
    return args + list(entry.get("extra_args", []))


def hyperqwen_command(entry, model_path, ctx, hyperqwen_root, python=None, port=None):
    """HyperQwen's own launcher (single-user/ or batch/start_qwen.sh) via colab_llm_hub.hyperqwen_launch,
    which first runs HyperQwen's CPU prepare scripts on a local copy of the downloaded checkpoint."""
    return [python or sys.executable, "-m", "colab_llm_hub.hyperqwen_launch",
            "--root", str(hyperqwen_root), "--model", str(model_path),
            "--port", str(port or PORTS["hyperqwen"]), "--ctx", str(int(ctx)),
            "--mode", entry.get("hyperqwen_mode", "single"), "--spec", entry.get("spec", "mtp"),
            "--thinking", "on" if entry.get("thinking") else "off"]


def ninfer_command(entry, model_path, ctx, ninfer_bin, port=None):
    """NInfer's OpenAI-compatible server. model_path is the .ninfer entry file (parts sit beside it)."""
    args = [str(ninfer_bin), str(model_path),
            "--host", "127.0.0.1", "--port", str(port or PORTS["ninfer"]),
            "--model-id", entry["id"], "--max-context", str(int(ctx)),
            "--kv-dtype", entry.get("kv_dtype", "fp8")]
    if not entry.get("thinking"):
        args.append("--no-thinking")    # server default only: a request can still set enable_thinking
    if entry.get("spec") and entry["spec"] != "none":
        args += ["--spec", entry["spec"], "--draft-tokens", str(entry.get("draft_tokens", 3))]
        if entry.get("proposal_head"):
            args += ["--proposal-head", entry["proposal_head"]]
    return args + list(entry.get("extra_args", []))


def _walk_names(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            out.append(os.path.join(dirpath, name))
    return out


def command(entry, model_path, ctx, bins, port=None):
    if entry["backend"] == "llama":
        return llama_command(entry, model_path, ctx, bins["llama"], port)
    if entry["backend"] == "diffusers":
        return diffusion_command(entry, model_path, ctx, bins.get("python"), port)
    if entry["backend"] == "sglang":
        return sglang_command(entry, model_path, ctx, bins.get("sglang"), port)
    if entry["backend"] == "strata":
        return strata_command(entry, model_path, ctx, bins.get("strata", "strata"), port,
                              data_dir=bins.get("strata_data"))
    if entry["backend"] == "maya":
        return maya_command(entry, model_path, ctx, bins.get("maya", "/content/project-maya"), port,
                            models_dir=bins.get("maya_models"))
    if entry["backend"] == "tensorfold":
        return tensorfold_command(entry, model_path, ctx,
                                  bins.get("tensorfold", "tensorfold-native"), port)
    if entry["backend"] == "comfyui":
        return comfyui_command(entry, model_path, ctx,
                               bins.get("comfyui", "/content/ComfyUI"),
                               bins.get("python"), port)
    if entry["backend"] == "sparsa":
        return sparsa_command(entry, model_path, ctx,
                              bins.get("sparsa", "/content/Sparsa"),
                              bins.get("python"), port)
    if entry["backend"] == "hyperqwen":
        return hyperqwen_command(entry, model_path, ctx, bins.get("hyperqwen", "/content/HyperQwen"),
                                 bins.get("python"), port)
    if entry["backend"] == "ninfer":
        return ninfer_command(entry, model_path, ctx,
                              bins.get("ninfer", "/content/ninfer/apps/ninfer-serve"), port)
    return vllm_command(entry, model_path, ctx, bins.get("vllm", "vllm"), port)


def env_for(entry, bins=None):
    env = dict(os.environ)
    if entry["backend"] == "hyperqwen":
        # No key: HyperQwen then binds 127.0.0.1 only, and the hub in front of it does the auth.
        env.pop("VLLM_API_KEY", None)
        env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", str(READY_TIMEOUT))
    if entry["backend"] == "ninfer":
        # The cached build bundles libcudart 13, liburing and libcurl next to apps/ in lib/.
        ninfer_bin = (bins or {}).get("ninfer", "/content/ninfer/apps/ninfer-serve")
        lib = os.path.join(os.path.dirname(os.path.dirname(str(ninfer_bin))), "lib")
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    if entry["backend"] in ("vllm", "sglang"):
        env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", str(READY_TIMEOUT))
        env["HF_HUB_OFFLINE"] = "1"                      # serve the local copy; never re-download
    if entry["backend"] in ("diffusers", "comfyui"):
        env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if entry["backend"] == "tensorfold":
        env.setdefault("TENSORFOLD_NO_UPDATE_CHECK", "1")
        env.setdefault("TENSORFOLD_NO_LIVE", "1")
        env["HF_HUB_OFFLINE"] = "1"
    if entry["backend"] in ("strata", "maya", "sparsa"):
        # Long first-start downloads; keep HF token available for gated packs.
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return env


def serving(port, model_id, timeout=3, backend=None):
    """True when the engine answers and lists the model (or any model for Strata/Maya)."""
    try:
        with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
            data = json.load(r).get("data", [])
            ids = {m.get("id") for m in data}
            if model_id in ids:
                return True
            if backend in ALIAS_READY_BACKENDS and data:
                return True
            return False
    except (OSError, ValueError):
        return False


class Engine:
    """One running engine process."""

    def __init__(self, entry, ctx, proc, port, log_path):
        self.entry, self.ctx, self.proc, self.port, self.log_path = entry, ctx, proc, port, log_path

    @property
    def alive(self):
        return self.proc.poll() is None

    def stop(self, grace=30):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + grace
            while self.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.5)
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=30)


def launch(entry, model_path, ctx, bins, log_dir, port=None, popen=subprocess.Popen):
    port = port or PORTS[entry["backend"]]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{entry['backend']}-{entry['id']}.log")
    with open(log_path, "ab") as log:                   # the child keeps its own copy of the descriptor
        proc = popen(command(entry, model_path, ctx, bins, port), env=env_for(entry, bins), stdin=subprocess.DEVNULL,
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return Engine(entry, ctx, proc, port, log_path)


def log_tail(path, lines=30):
    try:
        with open(path, "rb") as f:
            return b"".join(f.readlines()[-lines:]).decode(errors="replace")
    except OSError:
        return ""


def kind_for(entry):
    return "image" if entry["backend"] in IMAGE_BACKENDS else "chat"
