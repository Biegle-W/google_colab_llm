"""A tiny OpenAI-style engine used by the tests (stands in for llama-server / vllm)."""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, required=True)
p.add_argument("--model", required=True)
p.add_argument("--load-delay", type=float, default=0.0)
p.add_argument("--chunk-delay", type=float, default=0.0)
p.add_argument("--chunks", type=int, default=3)
p.add_argument("--exit-code", type=int, default=None, help="exit immediately with this status (a failed start)")
a = p.parse_args()
if a.exit_code is not None:
    print("fatal: could not load the model", flush=True)
    sys.exit(a.exit_code)
time.sleep(a.load_delay)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            return self.send(200, {"object": "list", "data": [{"id": a.model, "object": "model"}]})
        self.send(404, {"error": "no"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        if not req.get("stream"):
            return self.send(200, {"model": a.model, "choices": [{"message": {"role": "assistant",
                                    "content": f"reply from {a.model}"}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for i in range(a.chunks):
            time.sleep(a.chunk_delay)
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": f"{a.model}-{i} "}}]}).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
