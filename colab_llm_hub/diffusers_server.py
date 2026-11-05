"""OpenAI-compatible text-to-image server backed by Hugging Face Diffusers.

Started by the hub as a child process. Loads one pipeline from a local model directory
(already restored from Google Drive) and serves POST /v1/images/generations.
"""
import argparse
import base64
import io
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def load_pipeline(model_path, pipeline_name, dtype_name, device="cuda", quantization=None):
    import torch
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline

    if pipeline_name in ("unsupported_use_comfy_or_sglang", "unsupported"):
        raise RuntimeError(
            "This catalog entry is stored on Drive for ComfyUI / SGLang diffusion; "
            "the Diffusers OpenAI image server cannot start it. See ENGINE_MATRIX.md."
        )
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    loaders = {
        "AutoPipelineForText2Image": AutoPipelineForText2Image,
        "DiffusionPipeline": DiffusionPipeline,
    }
    cls = loaders.get(pipeline_name)
    if cls is None:
        # Named class from diffusers (e.g. QwenImage21Pipeline, FluxPipeline).
        import diffusers
        cls = getattr(diffusers, pipeline_name)
    kwargs = {"dtype": dtype, "local_files_only": True}
    # Prefer torch_dtype for older Diffusers; newer QwenImage21 uses dtype=.
    try:
        if quantization == "nf4":
            from diffusers.quantizers import PipelineQuantizationConfig
            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                              "bnb_4bit_compute_dtype": torch.float16},
                components_to_quantize=["transformer", "text_encoder"],
            )
            kwargs["low_cpu_mem_usage"] = True
            kwargs["device_map"] = "cuda"
            pipe = cls.from_pretrained(str(model_path), **kwargs)
        else:
            try:
                pipe = cls.from_pretrained(str(model_path), **kwargs)
            except TypeError:
                kwargs.pop("dtype", None)
                pipe = cls.from_pretrained(str(model_path), torch_dtype=dtype, local_files_only=True)
            pipe = pipe.to(device)
    except Exception:
        # Fall back to the classic torch_dtype path without quantization.
        pipe = cls.from_pretrained(str(model_path), torch_dtype=dtype, local_files_only=True)
        pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    return pipe


def clamp_size(width, height, max_edge):
    width = int(width or max_edge)
    height = int(height or max_edge)
    scale = min(1.0, max_edge / max(width, height))
    width = max(64, int(round(width * scale / 8) * 8))
    height = max(64, int(round(height * scale / 8) * 8))
    return width, height


def generate(pipe, prompt, *, negative_prompt=None, width=None, height=None, steps=None, guidance=None,
             seed=None, max_edge=1024, default_steps=28, default_guidance=7.5, n=1):
    import torch
    width, height = clamp_size(width, height, max_edge)
    steps = int(steps or default_steps)
    guidance = float(default_guidance if guidance is None else guidance)
    n = max(1, min(int(n or 1), 4))
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    kwargs = {"prompt": prompt, "num_inference_steps": steps, "width": width, "height": height,
              "num_images_per_prompt": n, "generator": generator}
    # Some pipelines (FLUX schnell) ignore or dislike negative/guidance; pass only when useful.
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if guidance is not None and guidance > 0:
        kwargs["guidance_scale"] = guidance
    try:
        result = pipe(**kwargs)
    except TypeError:
        kwargs.pop("negative_prompt", None)
        if guidance == 0:
            kwargs.pop("guidance_scale", None)
        result = pipe(**kwargs)
    images = []
    for img in result.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return images, width, height, steps


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
                                                      "owned_by": "colab-diffusers"}]})
            self.error_json(404, "Unknown endpoint.", "not_found")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/v1/images/generations":
                return self.error_json(404, "Only /v1/images/generations is exposed.", "not_found")
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
            try:
                images, width, height, steps = generate(
                    state["pipe"], prompt,
                    negative_prompt=data.get("negative_prompt"),
                    width=data.get("width") or (data.get("size", "x").split("x")[0] if "x" in str(data.get("size", "")) else None),
                    height=data.get("height") or (data.get("size", "x").split("x")[1] if "x" in str(data.get("size", "")) else None),
                    steps=data.get("steps") or data.get("num_inference_steps"),
                    guidance=data.get("guidance_scale", data.get("guidance")),
                    seed=data.get("seed"),
                    max_edge=state["max_edge"],
                    default_steps=state["default_steps"],
                    default_guidance=state["default_guidance"],
                    n=data.get("n", 1),
                )
            except Exception as e:
                traceback.print_exc()
                return self.error_json(500, f"Generation failed: {e}", "generation_failed")
            created = int(time.time())
            self.send_json(200, {
                "created": created,
                "model": state["model_id"],
                "data": [{"b64_json": b, "revised_prompt": None} for b in images],
                "size": f"{width}x{height}",
                "steps": steps,
            })

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(prog="colab_llm_hub.diffusers_server")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--port", type=int, default=30090)
    p.add_argument("--max-edge", type=int, default=1024)
    p.add_argument("--pipeline", default="AutoPipelineForText2Image")
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--quantization", default="", choices=("", "nf4"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    a.quantization = a.quantization or None

    print(f"[diffusers] loading {a.model_id} from {a.model_path} "
          f"({a.pipeline}, {a.dtype}, quant={a.quantization})", flush=True)
    pipe = load_pipeline(a.model_path, a.pipeline, a.dtype, device=a.device, quantization=a.quantization)
    state = {"pipe": pipe, "model_id": a.model_id, "max_edge": a.max_edge,
             "default_steps": a.steps, "default_guidance": a.guidance}
    print(f"[diffusers] ready on 127.0.0.1:{a.port} (max edge {a.max_edge})", flush=True)
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
