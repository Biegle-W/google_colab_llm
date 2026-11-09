"""OpenAI-compatible text-to-image (and text-to-video) server backed by ComfyUI.

Starts ComfyUI as a child process against a Drive-restored model directory, then exposes
POST /v1/images/generations (or, for a video model, POST /v1/videos/generations: prompt, size, seconds;
the MP4 comes back as b64_mp4) by queueing a stock API workflow graph.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
WORKFLOWS = Path(__file__).with_name("workflows")


FOLDERS = ("  checkpoints: checkpoints\n"
           "  diffusion_models: diffusion_models\n"
           "  text_encoders: text_encoders\n"
           "  clip: text_encoders\n"
           "  vae: vae\n"
           "  loras: loras\n"
           "  unet: diffusion_models\n")


def write_extra_model_paths(model_path, dest):
    """Point ComfyUI at the hub's restored model tree (diffusion_models / text_encoders / vae).

    Comfy-Org "Repackaged" repos (e.g. Wan 2.2) keep the same folders under split_files/, so that
    layout is registered as well.
    """
    root = Path(model_path).resolve()
    text = f"colab_hub:\n  base_path: {root}\n" + FOLDERS
    if (root / "split_files").is_dir():
        text += f"colab_hub_split_files:\n  base_path: {root / 'split_files'}\n" + FOLDERS
    Path(dest).write_text(text)
    return dest


def load_workflow(name):
    path = WORKFLOWS / name
    if not path.is_file():
        raise FileNotFoundError(f"workflow not found: {path}")
    return json.loads(path.read_text())


LOADER_FOLDERS = {"UNETLoader": ("unet_name", "diffusion_models"), "CLIPLoader": ("clip_name", "text_encoders"),
                  "VAELoader": ("vae_name", "vae"), "LoraLoaderModelOnly": ("lora_name", "loras")}


def bind_files(graph, model_path):
    """Point the workflow's loaders at the files this model's folder holds.

    One workflow serves several quants (INT8 on an L4 or A100, BF16, NVFP4 on a G4): each catalog entry downloads
    one file per role, so a loader whose file is missing takes the folder's file for that role (an audio VAE
    loader the audio VAE). A LoRA loader with no LoRA in the folder is bypassed."""
    root = Path(model_path)
    names = {}
    for folder in ("diffusion_models", "text_encoders", "vae", "loras"):
        found = []
        for base in (root, root / "split_files"):
            d = base / folder
            if d.is_dir():
                found += sorted(p.name for p in d.iterdir() if p.suffix in (".safetensors", ".gguf", ".pt"))
        names[folder] = found
    for nid, node in list(graph.items()):
        kind = node.get("class_type")
        if kind not in LOADER_FOLDERS:
            continue
        key, folder = LOADER_FOLDERS[kind]
        inputs = node.setdefault("inputs", {})
        have = names[folder]
        if inputs.get(key) in have:
            continue
        if kind == "VAELoader":
            audio = "audio" in str(inputs.get(key, ""))
            have = [n for n in have if ("audio" in n) == audio]
        if have:
            inputs[key] = have[0]
        elif kind == "LoraLoaderModelOnly":
            source = inputs.get("model")
            del graph[nid]
            for other in graph.values():
                for k, v in other.get("inputs", {}).items():
                    if isinstance(v, list) and v and v[0] == nid:
                        other["inputs"][k] = source
    return graph


def video_length(seconds, fps=24):
    """Frames for `seconds` of video on MiniMax H3's 17k+5 grid (the official template's formula)."""
    n = max(5, round(float(seconds) * fps))
    return n + (5 - n % 17) % 17


def apply_prompt(graph, prompt, *, negative="", seed=None, width=None, height=None, steps=None, seconds=None):
    """Fill prompt / seed / size into known node classes without depending on node ids."""
    seed = int(seed if seed is not None else (time.time_ns() % (2**31 - 1)))
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        kind = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if kind in ("CLIPTextEncode", "TextEncodeQwenImage21", "CLIPTextEncodeFlux"):
            if "prompt" in inputs:
                inputs["prompt"] = prompt
            if "text" in inputs and not inputs.get("_negative"):
                # Heuristic: empty / negative-looking nodes keep negative; others get the prompt.
                if inputs.get("text") in ("", None) or inputs.get("_role") == "negative":
                    inputs["text"] = negative or ""
                else:
                    inputs["text"] = prompt
            if "negative_prompt" in inputs:
                inputs["negative_prompt"] = negative or ""
            if width and "resolution" in inputs:
                inputs["resolution"] = int(max(width, height or width))
        if kind in ("RandomNoise", "KSampler", "SamplerCustom", "SamplerCustomAdvanced"):
            if "noise_seed" in inputs:
                inputs["noise_seed"] = seed
            if "seed" in inputs:
                inputs["seed"] = seed
            if steps is not None and "steps" in inputs:
                inputs["steps"] = int(steps)
        if kind == "MiniMaxH3ImageToVideo":
            inputs["prompt"] = prompt
            if width:
                inputs["width"] = int(width)
            if height:
                inputs["height"] = int(height)
            if seconds:
                inputs["length"] = video_length(seconds)
        if kind == "BasicScheduler" and steps is not None:
            inputs["steps"] = int(steps)
        if kind == "SaveVideo":
            inputs["filename_prefix"] = f"video/colab-{seed}"
        if kind in ("EmptyLatentImage", "EmptyFlux2LatentImage", "EmptySD3LatentImage"):
            if width:
                inputs["width"] = int(width)
            if height:
                inputs["height"] = int(height)
        if kind == "SaveImage":
            inputs["filename_prefix"] = f"colab-{seed}"
    return graph, seed


def wait_comfy(port, timeout=180):
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/system_stats"
    while time.time() < deadline:
        try:
            with opener.open(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (OSError, urllib.error.URLError, ValueError):
            time.sleep(1)
    return False


def queue_prompt(comfy_port, graph, client_id):
    body = json.dumps({"prompt": graph, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{comfy_port}/prompt", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with opener.open(req, timeout=60) as r:
            return json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:              # ComfyUI explains a rejected graph in the body (node_errors)
        raise RuntimeError(f"ComfyUI rejected the workflow ({e.code}): {e.read()[:2000].decode(errors='replace')}")


def wait_history(comfy_port, prompt_id, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{comfy_port}/history/{prompt_id}", timeout=10) as r:
                data = json.load(r)
                if prompt_id in data:
                    return data[prompt_id]
        except (OSError, urllib.error.URLError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {timeout}s")


def collect_images(comfy_port, history, output_dir):
    """Return PNG bytes for every SaveImage output in the history."""
    images = []
    outputs = history.get("outputs") or {}
    for node_out in outputs.values():
        for img in node_out.get("images") or []:
            filename = img["filename"]
            subfolder = img.get("subfolder") or ""
            img_type = img.get("type") or "output"
            # Prefer reading from ComfyUI's output folder directly when available.
            local = Path(output_dir) / subfolder / filename if img_type == "output" else None
            if local and local.is_file():
                images.append(local.read_bytes())
                continue
            qs = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": img_type})
            with opener.open(f"http://127.0.0.1:{comfy_port}/view?{qs}", timeout=60) as r:
                images.append(r.read())
    return images


# Late import for urlencode without requiring it at module import time for tests.
import urllib.parse  # noqa: E402


def clamp_size(width, height, max_edge, multiple=8):
    width = int(width or max_edge)
    height = int(height or max_edge)
    scale = min(1.0, max_edge / max(width, height))
    width = max(64, int(round(width * scale / multiple) * multiple))
    height = max(64, int(round(height * scale / multiple) * multiple))
    return width, height


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(self.command, self.path.split("?", 1)[0], flush=True)

        def send_json(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def error_json(self, status, message, code):
            self.send_json(status, {"error": {"message": message, "code": code, "type": "backend_error"}})

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/v1/models"):
                return self.send_json(200, {"object": "list",
                                            "data": [{"id": state["model_id"], "object": "model",
                                                      "owned_by": "colab-comfyui",
                                                      "kind": state.get("kind", "image")}]})
            self.error_json(404, "Unknown endpoint.", "not_found")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            video = state.get("kind") == "video"
            want = "/v1/videos/generations" if video else "/v1/images/generations"
            if path != want:
                return self.error_json(404, f"This engine serves {want} only.", "not_found")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length))
            except (ValueError, OSError):
                return self.error_json(400, "Invalid JSON request.", "bad_request")
            prompt = data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return self.error_json(400, "A text prompt is required.", "bad_request")
            model = data.get("model") or state["model_id"]
            if model != state["model_id"]:
                return self.error_json(404, f"This engine serves {state['model_id']}, not {model}.", "model_not_found")
            size = str(data.get("size") or "")
            width = data.get("width")
            height = data.get("height")
            if "x" in size:
                try:
                    width, height = (int(x) for x in size.lower().split("x", 1))
                except ValueError:
                    pass
            if video and not width and "x" not in size:
                width, height = 832, 480                         # 16:9 at ~0.4 MP, the official template's size
            width, height = clamp_size(width, height, state["max_edge"], multiple=32 if video else 8)
            steps = data.get("steps") or data.get("num_inference_steps") or state["default_steps"]
            seconds = min(float(data.get("seconds") or data.get("duration") or 5), 15.0) if video else None
            try:
                graph = json.loads(json.dumps(state["workflow"]))  # deep copy
                graph = bind_files(graph, state["model_path"])
                graph, seed = apply_prompt(
                    graph, prompt, negative=data.get("negative_prompt") or "",
                    seed=data.get("seed"), width=width, height=height, steps=steps, seconds=seconds)
                prompt_id = queue_prompt(state["comfy_port"], graph, state["client_id"])
                history = wait_history(state["comfy_port"], prompt_id, timeout=3600 if video else 900)
                pngs = collect_images(state["comfy_port"], history, state["output_dir"])
                if not pngs:
                    return self.error_json(500, f"ComfyUI finished without {'video' if video else 'image'} outputs.",
                                           "generation_failed")
            except Exception as e:
                traceback.print_exc()
                return self.error_json(500, f"Generation failed: {e}", "generation_failed")
            if video:
                return self.send_json(200, {
                    "created": int(time.time()), "model": state["model_id"],
                    "data": [{"b64_mp4": base64.b64encode(pngs[0]).decode("ascii")}],
                    "size": f"{width}x{height}", "seconds": seconds, "steps": int(steps), "seed": seed})
            self.send_json(200, {
                "created": int(time.time()),
                "model": state["model_id"],
                "data": [{"b64_json": base64.b64encode(p).decode("ascii"), "revised_prompt": None}
                         for p in pngs[: max(1, min(int(data.get("n", 1)), 4))]],
                "size": f"{width}x{height}",
                "steps": int(steps),
                "seed": seed,
            })

    return Handler


def stop_stale_comfy(comfy_port, sleep=time.sleep):
    """A ComfyUI left on our port (an earlier model's, from a wrapper that was killed) would answer in place of
    the new one, with the wrong models: stop it before starting ours."""
    if not wait_comfy(comfy_port, timeout=1):
        return False
    print(f"[comfyui] stopping a ComfyUI already on :{comfy_port} (left by an earlier model)", flush=True)
    subprocess.run(["pkill", "-TERM", "-f", f"main.py --listen 127.0.0.1 --port {int(comfy_port)}"], capture_output=True)
    for _ in range(30):
        if not wait_comfy(comfy_port, timeout=1):
            return True
        sleep(1)
    subprocess.run(["pkill", "-KILL", "-f", f"main.py --listen 127.0.0.1 --port {int(comfy_port)}"], capture_output=True)
    return True


def start_comfy(comfy_root, model_path, comfy_port, log_path):
    stop_stale_comfy(comfy_port)
    extra = Path(comfy_root) / "extra_model_paths.yaml"
    write_extra_model_paths(model_path, extra)
    output_dir = Path(comfy_root) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "main.py",
           "--listen", "127.0.0.1", "--port", str(comfy_port),
           "--disable-auto-launch",
           "--extra-model-paths-config", str(extra)]
    log = open(log_path, "ab")
    # Same process group as this wrapper, so the hub's stop (a signal to the group) ends ComfyUI too.
    proc = subprocess.Popen(cmd, cwd=comfy_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not wait_comfy(comfy_port, timeout=300):
        proc.terminate()
        raise RuntimeError(f"ComfyUI did not become ready on :{comfy_port}; see {log_path}")
    return proc, str(output_dir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="notebook_code.comfyui_server")
    p.add_argument("--comfy-root", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--port", type=int, default=30095)
    p.add_argument("--comfy-port", type=int, default=8188)
    p.add_argument("--max-edge", type=int, default=1024)
    p.add_argument("--workflow", default="qwen-image-2.1-turbo.api.json")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--kind", default="image", choices=("image", "video"))
    a = p.parse_args(argv)

    if not Path(a.comfy_root, "main.py").is_file():
        raise SystemExit(f"ComfyUI main.py not found under {a.comfy_root}")
    workflow = load_workflow(a.workflow)
    log_path = os.path.join(a.comfy_root, f"comfyui-{a.model_id}.log")
    print(f"[comfyui] starting ComfyUI for {a.model_id} (workflow {a.workflow})", flush=True)
    proc, output_dir = start_comfy(a.comfy_root, a.model_path, a.comfy_port, log_path)
    state = {
        "model_id": a.model_id, "max_edge": a.max_edge, "default_steps": a.steps,
        "default_guidance": a.guidance, "workflow": workflow, "comfy_port": a.comfy_port,
        "client_id": str(uuid.uuid4()), "output_dir": output_dir, "kind": a.kind, "model_path": a.model_path,
        "proc": proc,
    }
    print(f"[comfyui] ready on 127.0.0.1:{a.port} (ComfyUI UI :{a.comfy_port})", flush=True)
    import signal

    def on_term(signum, frame):                  # SIGTERM skips `finally`: stop ComfyUI explicitly
        if proc.poll() is None:
            proc.terminate()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, on_term)
    try:
        ThreadingHTTPServer.daemon_threads = True
        ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state)).serve_forever()
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
