"""Authenticated OpenAI-compatible hub: one endpoint for every catalog model that fits the GPU.

A request naming a model is proxied to its engine. If another model (or none) is loaded, a background
worker downloads/copies the files, waits for in-flight replies on the old model, stops it, starts the new
engine and waits until it answers. Streaming clients get SSE comment keep-alives while this happens;
non-streaming clients wait a bounded time and then receive 503 + Retry-After while the load continues.
Switching logic follows the local model hub this stack is modelled on: replies are never cut off, and
switches are serialized.
"""
import hmac
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from . import backends, catalog, gpu

MAX_BODY = 32 * 1024 * 1024
KEEPALIVE = 10            # seconds between SSE comment keep-alives
DRAIN_TIMEOUT = 900       # wait this long for in-flight replies before refusing a switch
STREAM_WAIT = 7200        # a streaming request may wait this long for a model (large downloads)
NONSTREAM_WAIT = 85       # stay under Cloudflare's ~100 s idle limit
POLL = 1                  # seconds between readiness checks while an engine starts
ERROR_TTL = 20            # repeat a failed load's error for this long instead of retrying at once


class HubError(Exception):
    def __init__(self, status, message, code, retry_after=None):
        super().__init__(message)
        self.status, self.code, self.retry_after = status, code, retry_after


class Hub:
    def __init__(self, entries, vram_gb, store, bins, log_dir, ctx_overrides=None, launch=backends.launch,
                 serving=backends.serving, gpu_used=gpu.used_mib, sleep=time.sleep, now=time.monotonic,
                 ready_timeout=backends.READY_TIMEOUT, gpu_name="", log=print):
        self.vram_gb, self.gpu_name = vram_gb, gpu_name
        self.all_entries = catalog.by_id(entries)
        self.entries = catalog.by_id(catalog.visible(entries, vram_gb))
        self.store, self.bins, self.log_dir = store, bins, log_dir
        self.ctx_overrides = ctx_overrides or {}
        self._launch, self._serving, self._gpu_used = launch, serving, gpu_used
        self.sleep, self.now, self.ready_timeout, self.log = sleep, now, ready_timeout, log
        self.cond = threading.Condition()
        self.engine = None        # running backends.Engine
        self.active = None        # model id that is ready to serve
        self.target = None        # model id the loader should load next
        self.loading = None       # model id being loaded right now
        self.phase = ""           # human-readable step of the current load
        self.inflight = 0
        self.switching = False
        self.errors = {}          # model id -> (message, code, time)
        self.stopping = False
        self.worker = threading.Thread(target=self._loader_loop, name="hub-loader", daemon=True)
        self.worker.start()

    # ---- model info ------------------------------------------------------------------------
    def context(self, model_id):
        return catalog.context_for(self.entries[model_id], self.vram_gb, self.ctx_overrides.get(model_id))

    def describe(self, model_id):
        e = self.entries[model_id]
        st = self.store.state(e)
        return {"id": e["id"], "object": "model", "created": 0, "owned_by": "colab-" + e["backend"],
                "name": e["name"], "backend": e["backend"], "context_length": self.context(e["id"]),
                "size_gb": e["size_gb"], "loaded": self.active == e["id"], "loading": self.loading == e["id"],
                "downloaded": st["on_drive"], "experimental": bool(e.get("experimental"))}

    def status(self):
        with self.cond:
            return {"gpu": self.gpu_name, "vram_gb": self.vram_gb, "active": self.active, "loading": self.loading,
                    "phase": self.phase, "inflight": self.inflight, "switching": self.switching,
                    "errors": {k: {"message": v[0], "code": v[1]} for k, v in self.errors.items()},
                    "downloads": dict(self.store.progress),
                    "local_free_gb": round(self.store.free_gb(), 1), "models": list(self.entries)}

    # ---- request side ----------------------------------------------------------------------
    def acquire(self, model_id, progress=lambda: None, timeout=NONSTREAM_WAIT):
        """Reserve the engine for one request, loading the model if needed. Returns the engine's port."""
        if model_id not in self.entries:
            raise HubError(404, "Unknown model, or it does not fit this GPU. Use GET /v1/models.", "model_not_found")
        deadline = self.now() + timeout
        with self.cond:
            while True:
                if self.stopping:
                    raise HubError(503, "The hub is shutting down.", "hub_stopping")
                if self.active == model_id and self.engine is not None and not self.engine.alive:
                    self.active = None                             # the engine crashed; reload on demand
                if self.active == model_id and not self.switching:
                    self.inflight += 1
                    return self.engine.port
                err = self.errors.get(model_id)
                if err and self.loading != model_id and self.target != model_id:
                    if self.now() - err[2] < ERROR_TTL:
                        raise HubError(503, err[0], err[1])
                    self.errors.pop(model_id, None)
                if self.target is None and self.loading is None:
                    self.target = model_id
                    self.cond.notify_all()
                remaining = deadline - self.now()
                if remaining <= 0:
                    raise HubError(503, f"{model_id} is still loading ({self.phase or 'waiting'}); retry shortly.",
                                   "model_loading", retry_after=15)
                self.cond.wait(min(1.0, remaining))
                progress()

    def release(self):
        with self.cond:
            self.inflight -= 1
            self.cond.notify_all()

    # ---- loader ----------------------------------------------------------------------------
    def _loader_loop(self):
        while True:
            with self.cond:
                while self.target is None and not self.stopping:
                    self.cond.wait()
                if self.stopping:
                    return
                model, self.target, self.loading = self.target, None, self.target
            error = None
            try:
                self._load(model)
            except HubError as e:
                error = e
            except Exception as e:                                  # download failures, disk full, launch errors
                error = HubError(503, f"{model} could not be loaded: {e}", "backend_start_failed")
            with self.cond:
                self.loading, self.phase, self.switching = None, "", False
                if error:
                    self.log(f"[hub] load of {model} failed: {error}")
                    self.errors[model] = (str(error), error.code, self.now())
                else:
                    self.errors.pop(model, None)
                self.cond.notify_all()

    def _set_phase(self, text):
        with self.cond:
            self.phase = text

    def _load(self, model_id):
        entry = self.entries[model_id]
        # 1. Get the files while the current model keeps serving.
        self._set_phase("getting model files")
        protect = [self.entries[self.active]] if self.active in self.entries else []
        path = self.store.ensure_local(entry, protect=protect)
        ctx = self.context(model_id)
        # 2. Wait for in-flight replies, then stop the old engine.
        self._set_phase("waiting for the current model's replies to finish")
        with self.cond:
            self.switching = True
            deadline = self.now() + DRAIN_TIMEOUT
            while self.inflight > 0:
                if self.now() >= deadline:
                    raise HubError(503, "Another model is still answering; retry when it finishes.", "backend_busy")
                self.cond.wait(1.0)
            old, self.engine, self.active = self.engine, None, None
        if old is not None:
            self._set_phase("stopping the previous model")
            old.stop()
            self._wait_gpu_free()
        # 3. Start the new engine and wait until it answers.
        self._set_phase("starting the engine")
        engine = self._launch(entry, path, ctx, self.bins, self.log_dir)
        deadline = self.now() + self.ready_timeout
        while True:
            if self.stopping:
                engine.stop()
                raise HubError(503, "The hub is shutting down.", "hub_stopping")
            if self._serving(engine.port, model_id):
                break
            if not engine.alive:
                tail = backends.log_tail(engine.log_path, 15).strip().splitlines()
                raise HubError(503, f"{entry['backend']} could not start {model_id}: "
                                    f"{tail[-1] if tail else 'see ' + engine.log_path}", "backend_start_failed")
            if self.now() >= deadline:
                engine.stop()
                raise HubError(504, f"{model_id} did not become ready in {self.ready_timeout // 60} minutes.",
                               "backend_start_timeout")
            self._set_phase(f"loading into GPU memory ({int(deadline - self.now())} s left before timeout)")
            self.sleep(POLL)
        with self.cond:
            self.engine, self.active = engine, model_id
            self.log(f"[hub] {model_id} is ready (context {ctx})")

    def _wait_gpu_free(self, limit_mib=2000, seconds=90):
        for _ in range(seconds):
            used = self._gpu_used()
            if used is None or used < limit_mib:
                return
            self.sleep(1)

    # ---- admin -----------------------------------------------------------------------------
    def prefetch(self, model_id):
        if model_id not in self.entries:
            raise HubError(404, "Unknown model, or it does not fit this GPU.", "model_not_found")
        entry = self.entries[model_id]
        protect = [self.entries[self.active]] if self.active in self.entries else []

        def run():
            try:
                self.store.prefetch(entry, protect=protect)
            except Exception as e:
                self.log(f"[hub] prefetch of {model_id} failed: {e}")
        threading.Thread(target=run, name=f"prefetch-{model_id}", daemon=True).start()

    def unload(self):
        with self.cond:
            if self.inflight:
                raise HubError(409, "A request is still being answered.", "backend_busy")
            old, self.engine, self.active = self.engine, None, None
        if old is not None:
            old.stop()

    def shutdown(self):
        with self.cond:
            self.stopping = True
            self.cond.notify_all()
        if self.engine is not None:
            self.engine.stop()


def make_handler(hub, token):
    token = token.encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):                 # never log bodies or credentials
            print(self.command, urlsplit(self.path).path, flush=True)

        def send_json(self, status, value, headers=None):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def error_json(self, status, message, code, retry_after=None):
            headers = {"Retry-After": retry_after} if retry_after else None
            if status == 401:
                headers = dict(headers or {}, **{"WWW-Authenticate": 'Bearer realm="colab-llm-hub"'})
            self.send_json(status, {"error": {"message": message, "code": code,
                                              "type": "invalid_request_error" if status < 500 else "backend_error"}}, headers)

        def authorized(self):
            header = self.headers.get("Authorization", "")
            supplied = header[7:].encode() if header.startswith("Bearer ") else b""
            if not supplied or not hmac.compare_digest(supplied, token):
                self.error_json(401, "A valid API key is required.", "invalid_api_key")
                return False
            return True

        # ---- GET ----
        def do_GET(self):
            if not self.authorized():
                return
            path = urlsplit(self.path).path
            if path == "/health":
                return self.send_json(200, {"status": "ok", "active": hub.active, "loading": hub.loading})
            if path == "/v1/hub/status":
                return self.send_json(200, hub.status())
            if path == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [hub.describe(i) for i in hub.entries]})
            if path.startswith("/v1/models/"):
                mid = unquote(path.removeprefix("/v1/models/"))
                if mid in hub.entries:
                    return self.send_json(200, hub.describe(mid))
                return self.error_json(404, "Unknown model.", "model_not_found")
            self.error_json(404, "Unknown endpoint.", "not_found")

        # ---- POST ----
        def read_json(self):
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("chunked request bodies are not supported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise OverflowError
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("incomplete body")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("body must be an object")
            return body, data

        def do_POST(self):
            if not self.authorized():
                return
            path = urlsplit(self.path).path
            try:
                body, data = self.read_json()
            except OverflowError:
                return self.error_json(413, "Request must be between 1 byte and 32 MiB.", "body_size")
            except (ValueError, OSError):
                return self.error_json(400, "Invalid JSON request.", "bad_request")
            self.close_connection = True
            if path == "/v1/hub/prefetch":
                try:
                    hub.prefetch(str(data.get("model", "")))
                except HubError as e:
                    return self.error_json(e.status, str(e), e.code)
                return self.send_json(202, {"status": "started", "model": data["model"]})
            if path == "/v1/hub/unload":
                try:
                    hub.unload()
                except HubError as e:
                    return self.error_json(e.status, str(e), e.code)
                return self.send_json(200, {"status": "unloaded"})
            if path not in ("/v1/chat/completions", "/v1/completions"):
                return self.error_json(404, "Only OpenAI chat/completion endpoints are exposed.", "not_found")
            model = data.get("model")
            if not isinstance(model, str):
                return self.error_json(400, "A model ID is required. Use GET /v1/models.", "bad_request")
            if data.get("stream"):
                return self.stream(model, path, body)
            try:
                port = hub.acquire(model, timeout=NONSTREAM_WAIT)
            except HubError as e:
                return self.error_json(e.status, str(e), e.code, e.retry_after)
            try:
                self.forward(port, path, body)
            finally:
                hub.release()

        def forward(self, port, path, body):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=86400)
            try:
                conn.request("POST", path, body, {"Content-Type": "application/json"})
                resp = conn.getresponse()
                payload = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                self.error_json(503, "Engine connection failed; see /content/hub_logs.", "backend_unavailable")
            finally:
                conn.close()

        def stream(self, model, path, body):
            # Headers go out at once so the client (and Cloudflare) sees progress while a model loads.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            lock = threading.Lock()

            def write(chunk):
                with lock:
                    self.wfile.write(chunk)
                    self.wfile.flush()

            def event_error(status, message, code):
                write(b"data: " + json.dumps({"error": {"message": message, "code": code, "status": status}}).encode()
                      + b"\n\ndata: [DONE]\n\n")
            last = [time.monotonic()]

            def progress():
                if time.monotonic() - last[0] >= KEEPALIVE:
                    write(f": loading {model} - {hub.phase}\n\n".encode())
                    last[0] = time.monotonic()
            try:
                port = hub.acquire(model, progress, timeout=STREAM_WAIT)
            except HubError as e:
                return event_error(e.status, str(e), e.code)
            except OSError:
                return
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=86400)
            try:
                conn.request("POST", path, body, {"Content-Type": "application/json"})
                resp = conn.getresponse()
                if resp.status != 200:
                    payload = resp.read()
                    try:
                        message = json.loads(payload)["error"]["message"]
                    except (ValueError, KeyError, TypeError):
                        message = payload.decode(errors="replace")[:500]
                    return event_error(resp.status, message, "backend_error")
                while chunk := resp.read1(65536):
                    write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                try:
                    event_error(503, "Engine connection failed; see /content/hub_logs.", "backend_unavailable")
                except OSError:
                    pass
            finally:
                conn.close()
                hub.release()

    return Handler


def make_server(hub, token, host="127.0.0.1", port=8000):
    ThreadingHTTPServer.daemon_threads = True
    return ThreadingHTTPServer((host, port), make_handler(hub, token))
