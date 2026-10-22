import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

from colab_llm_hub import backends, hub as hubmod
from colab_llm_hub.store import ModelStore

FAKE = os.path.join(os.path.dirname(__file__), "fake_engine.py")
KEY = "test-key-0123456789abcdef0123456789"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def entry(id, backend="llama", size=0.001, min_vram=10, **kw):
    e = {"id": id, "name": id.upper(), "backend": backend, "repo": f"org/{id}", "files": ["*"],
         "size_gb": size, "contexts": [[min_vram, 4096], [min_vram + 20, 8192]]}
    if backend == "llama":
        e["model_file"] = f"{id}.gguf"
    e.update(kw)
    return e


def fake_downloader(calls):
    def download(repo, patterns, dest, token=None):
        calls.append(repo)
        os.makedirs(dest, exist_ok=True)
        name = repo.split("/")[1]
        (dest / f"{name}.gguf").write_bytes(b"x" * 2048)
        (dest / "config.json").write_text("{}")
    return download


class HubCase(unittest.TestCase):
    """Starts a real hub server on an ephemeral port with fake engine processes."""
    vram = 24.0
    entries = None
    load_delay = 0.0
    chunk_delay = 0.0
    fail = set()                 # model ids whose engine exits at once

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hubtest-")
        self.downloads = []
        self.launches = []
        self.store = ModelStore(os.path.join(self.tmp, "drive"), os.path.join(self.tmp, "local"),
                                downloader=fake_downloader(self.downloads), min_free_gb=0, log=lambda *_: None)
        entries = self.entries or [entry("a"), entry("b"), entry("big", min_vram=60)]
        self.hub = hubmod.Hub(entries, self.vram, self.store, {"llama": "x"}, os.path.join(self.tmp, "logs"),
                              launch=self.launch, ready_timeout=20, log=lambda *_: None, gpu_name="fake")
        self.server = hubmod.make_server(self.hub, KEY, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.hub.shutdown()
        self.server.server_close()
        for e in self.launches:
            e.stop(grace=2)

    def launch(self, entry, path, ctx, bins, log_dir):
        port = free_port()
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(log_dir, f"{entry['id']}.log")
        args = [sys.executable, FAKE, "--port", str(port), "--model", entry["id"],
                "--load-delay", str(self.load_delay), "--chunk-delay", str(self.chunk_delay)]
        if entry["id"] in self.fail:
            args += ["--exit-code", "3"]
        with open(log, "ab") as f:
            proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        eng = backends.Engine(entry, ctx, proc, port, log)
        self.launches.append(eng)
        return eng

    # ---- client helpers ----
    def request(self, method, path, body=None, key=KEY, timeout=30):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        conn.request(method, path, json.dumps(body) if body is not None else None, headers)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        try:
            data = json.loads(data)
        except ValueError:
            pass
        return r.status, data, dict(r.getheaders())

    def chat(self, model, stream=False, **kw):
        return self.request("POST", "/v1/chat/completions",
                            {"model": model, "stream": stream, "messages": [{"role": "user", "content": "hi"}]}, **kw)

    def stream_raw(self, model, timeout=30):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        conn.request("POST", "/v1/chat/completions", json.dumps({"model": model, "stream": True,
                     "messages": [{"role": "user", "content": "hi"}]}),
                     {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        r = conn.getresponse()
        return r.status, r.read().decode(), conn
