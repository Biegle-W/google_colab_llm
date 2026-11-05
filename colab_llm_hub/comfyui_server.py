"""OpenAI-compatible text-to-image server backed by ComfyUI.

Starts ComfyUI as a child process against a Drive-restored model directory, then exposes
POST /v1/images/generations by queueing a stock API workflow graph.
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


def write_extra_model_paths(model_path, dest):
    """Point ComfyUI at the hub's restored model tree (diffusion_models / text_encoders / vae)."""
    root = Path(model_path).resolve()
    text = (
        "colab_hub:\n"
        f"  base_path: {root}\n"
        "  checkpoints: checkpoints\n"
        "  diffusion_models: diffusion_models\n"
        "  text_encoders: text_encoders\n"
        "  clip: text_encoders\n"
        "  vae: vae\n"
        "  loras: loras\n"
        "  unet: diffusion_models\n"
    )
    Path(dest).write_text(text)
    return dest


def load_workflow(name):
    path = WORKFLOWS / name
    if not path.is_file():
        raise FileNotFoundError(f"workflow not found: {path}")
    return json.loads(path.read_text())


def apply_prompt(graph, prompt, *, negative="", seed=None, width=None, height=None, steps=None):
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
    with opener.open(req, timeout=60) as r:
        return json.load(r)["prompt_id"]


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


def clamp_size(width, height, max_edge):
    width = int(width or max_edge)
    height = int(height or max_edge)
    scale = min(1.0, max_edge / max(width, height))
    width = max(64, int(round(width * scale / 8) * 8))
    height = max(64, int(round(height * scale / 8) * 8))
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
            if path != "/v1/images/generations":
                return self.error_json(404, "Only /v1/images/generations is exposed.", "not_found")
            if state.get("kind") == "video":
                return self.error_json(501, "Video workflows are prefetch-only on this Colab hub; "
                                       "open ComfyUI UI workflows for interactive video.", "unsupported")
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
            width, height = clamp_size(width, height, state["max_edge"])
            steps = data.get("steps") or data.get("num_inference_steps") or state["default_steps"]
            try:
                graph = json.loads(json.dumps(state["workflow"]))  # deep copy
                graph, seed = apply_prompt(
                    graph, prompt, negative=data.get("negative_prompt") or "",
                    seed=data.get("seed"), width=width, height=height, steps=steps)
                prompt_id = queue_prompt(state["comfy_port"], graph, state["client_id"])
                history = wait_history(state["comfy_port"], prompt_id)
                pngs = collect_images(state["comfy_port"], history, state["output_dir"])
                if not pngs:
                    return self.error_json(500, "ComfyUI finished without image outputs.", "generation_failed")
            except Exception as e:
                traceback.print_exc()
                return self.error_json(500, f"Generation failed: {e}", "generation_failed")
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


def start_comfy(comfy_root, model_path, comfy_port, log_path):
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
    proc = subprocess.Popen(cmd, cwd=comfy_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    if not wait_comfy(comfy_port, timeout=300):
        proc.terminate()
        raise RuntimeError(f"ComfyUI did not become ready on :{comfy_port}; see {log_path}")
    return proc, str(output_dir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="colab_llm_hub.comfyui_server")
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
        "client_id": str(uuid.uuid4()), "output_dir": output_dir, "kind": a.kind,
        "proc": proc,
    }
    print(f"[comfyui] ready on 127.0.0.1:{a.port} (ComfyUI UI :{a.comfy_port})", flush=True)
    try:
        ThreadingHTTPServer.daemon_threads = True
        ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state)).serve_forever()
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
