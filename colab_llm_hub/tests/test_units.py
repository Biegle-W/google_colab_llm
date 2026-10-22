import json
import os
import shutil
import tempfile
import time
import unittest

from colab_llm_hub import backends, catalog, gpu, install, keyfile, provider, tunnel
from colab_llm_hub.store import MARKER, ModelStore, StoreError, is_complete, write_marker
from .helpers import entry, fake_downloader


class Catalog(unittest.TestCase):
    def test_shipped_catalog_is_valid(self):
        entries = catalog.load()
        self.assertGreater(len(entries), 10)
        for e in entries:
            self.assertTrue(e["files"], e["id"])
            self.assertGreater(e["size_gb"], 1, e["id"])
            # The smallest context threshold must at least cover the weights.
            self.assertGreaterEqual(catalog.min_vram_gb(e), e["size_gb"], e["id"])

    def test_fit_and_context_by_gpu(self):
        by = catalog.by_id(catalog.load())
        e = by["qwen3.8-27b-q8"]
        self.assertFalse(catalog.fits(e, 24.2))                               # L4
        self.assertEqual(catalog.context_for(e, 42.9), 32768)                 # A100 40 GB
        self.assertEqual(catalog.context_for(e, 102.6), 65536)                # G4
        self.assertEqual(catalog.context_for(e, 42.9, override=8192), 8192)
        with self.assertRaises(catalog.CatalogError):
            catalog.context_for(e, 24.2)
        self.assertEqual(catalog.visible(catalog.load(), 16.1), [])           # T4: use the T4 notebook
        self.assertTrue(any(e["id"] == "ornith-1.5-122b-q4" for e in catalog.visible(catalog.load(), 102.6)))
        self.assertFalse(any(e["id"] == "ornith-1.5-122b-q4" for e in catalog.visible(catalog.load(), 42.9)))

    def test_engine_filter(self):
        entries = catalog.load()
        llama = catalog.for_engines(entries, ["llama"])
        self.assertTrue(llama and all(e["backend"] == "llama" for e in llama))
        self.assertEqual(len(catalog.for_engines(entries, ["llama", "vllm"])), len(entries))
        self.assertEqual(catalog.for_engines(entries, []), [])

    def test_validation_rejects_bad_entries(self):
        good = entry("a")
        for bad in ({**good, "backend": "tgi"}, {**good, "contexts": [[30, 8192], [20, 16384]]},
                    {k: v for k, v in good.items() if k != "repo"}):
            with self.assertRaises(catalog.CatalogError):
                catalog.validate([bad])
        with self.assertRaises(catalog.CatalogError):
            catalog.validate([good, good])


class Gpu(unittest.TestCase):
    def test_parse_smi(self):
        out = "NVIDIA L4, 23034, 8.9\nNVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0\n"
        g = gpu.parse_smi(out)
        self.assertEqual((g[0]["arch"], g[0]["vram_gb"]), ("89", 24.2))
        self.assertEqual((g[1]["arch"], g[1]["vram_gb"]), ("120", 102.6))

    def test_cmake_arch(self):
        self.assertEqual(gpu.cmake_arch("89", (12, 4)), "89")
        self.assertEqual(gpu.cmake_arch("120", (12, 4)), "90-virtual")
        self.assertEqual(gpu.cmake_arch("120", (12, 8)), "120")
        self.assertEqual(gpu.nvcc_version("Cuda compilation tools, release 12.5, V12.5.82"), (12, 5))
        self.assertIsNone(gpu.nvcc_version(""))


class Commands(unittest.TestCase):
    def test_llama_command(self):
        e = entry("m", backend="llama")
        args = backends.llama_command(e, "/models/m.gguf", 32768, "/bin/llama-server", port=1234)
        val = lambda flag: args[args.index(flag) + 1]          # boolean flags such as --jinja take no value
        self.assertEqual(args[0], "/bin/llama-server")
        self.assertEqual((val("-m"), val("--alias"), val("-c"), val("--port"), val("-ngl")),
                         ("/models/m.gguf", "m", "32768", "1234", "99"))
        self.assertEqual(val("--host"), "127.0.0.1")
        self.assertEqual((val("--flash-attn"), val("--cache-type-k"), val("--reasoning")), ("on", "q8_0", "off"))
        for needed in ("--jinja", "--flash-attn", "--cache-type-k", "--cache-ram", "--ctx-checkpoints", "--reasoning"):
            self.assertIn(needed, args)
        self.assertEqual(json.loads(val("--chat-template-kwargs")), {"enable_thinking": False})
        thinking = backends.llama_command({**e, "thinking": True, "extra_args": ["--x", "1"]}, "p", 1, "b")
        self.assertNotIn("--reasoning", thinking)
        self.assertEqual(thinking[-2:], ["--x", "1"])

    def test_vllm_command(self):
        e = entry("v", backend="vllm", tool_parser="qwen3_coder", reasoning_parser="qwen3", language_model_only=True)
        args = backends.vllm_command(e, "/models/v", 16384, "vllm", port=999, gpu_util=0.92)
        self.assertEqual(args[:3], ["vllm", "serve", "/models/v"])
        val = lambda flag: args[args.index(flag) + 1]
        self.assertEqual((val("--served-model-name"), val("--max-model-len"), val("--gpu-memory-utilization")),
                         ("v", "16384", "0.92"))
        self.assertEqual((val("--tool-call-parser"), val("--reasoning-parser")), ("qwen3_coder", "qwen3"))
        for needed in ("--enable-auto-tool-choice", "--language-model-only", "--default-chat-template-kwargs"):
            self.assertIn(needed, args)
        plain = backends.vllm_command(entry("p", backend="vllm"), "/m", 8192, "vllm")
        self.assertNotIn("--tool-call-parser", plain)
        self.assertNotIn("--language-model-only", plain)

    def test_env(self):
        self.assertEqual(backends.env_for(entry("v", backend="vllm"))["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("HF_HUB_OFFLINE", {k: v for k, v in backends.env_for(entry("l")).items()
                                            if os.environ.get(k) != v})


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="storetest-")
        self.calls = []
        self.store = ModelStore(self.tmp + "/drive", self.tmp + "/local", downloader=fake_downloader(self.calls),
                                min_free_gb=0, log=lambda *_: None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_download_saves_to_drive_then_later_session_copies_back(self):
        e = entry("a")
        path = self.store.ensure_local(e)
        self.assertTrue(path.is_file())
        self.store.wait_for_drive_copies()
        self.assertTrue(is_complete(self.store.drive_path(e)))
        # A new Colab session: empty local disk, same Drive.
        shutil.rmtree(self.tmp + "/local")
        again = ModelStore(self.tmp + "/drive", self.tmp + "/local", downloader=fake_downloader(self.calls),
                           min_free_gb=0, log=lambda *_: None)
        self.assertTrue(again.ensure_local(e).is_file())
        self.assertEqual(self.calls, ["org/a"])                      # downloaded exactly once

    def test_interrupted_copy_is_not_trusted(self):
        e = entry("a")
        self.store.ensure_local(e)
        self.store.wait_for_drive_copies()
        drive = self.store.drive_path(e)
        (drive / "a.gguf").write_bytes(b"short")                    # truncated file, marker still present
        self.assertFalse(is_complete(drive))
        shutil.rmtree(self.tmp + "/local")
        self.store.ensure_local(e)
        self.assertEqual(self.calls, ["org/a", "org/a"])             # fell back to a fresh download
        (drive / MARKER).unlink(missing_ok=True)
        self.assertFalse(is_complete(drive))

    def test_part_files_are_ignored(self):
        e = entry("a")
        d = self.store.local_path(e)
        d.mkdir(parents=True)
        (d / "a.gguf").write_bytes(b"x")
        (d / "b.gguf.part").write_bytes(b"x")
        write_marker(d, "org/a")
        self.assertEqual(list(json.loads((d / MARKER).read_text())["files"]), ["a.gguf"])

    def test_missing_model_file_is_an_error(self):
        e = entry("a", model_file="nothere.gguf")
        with self.assertRaises(StoreError):
            self.store.ensure_local(e)

    def test_lru_eviction_keeps_active_and_drive_copies(self):
        small = lambda i: entry(i, size=1.0)
        a, b, c = small("a"), small("b"), small("c")
        self.store.min_free_gb = 0
        for e in (a, b, c):
            self.store.ensure_local(e)
            time.sleep(0.05)
        self.store.wait_for_drive_copies()
        free = self.store.free_gb()
        # Pretend the disk only has room for one more model beyond what is there.
        self.store.free_gb = lambda: 0.5 if len(self.store.local_models()) > 1 else 100
        d = small("d")
        self.store.ensure_local(d, protect=[c])
        remaining = {p.name for p in self.store.local_models()}
        self.assertIn("c", remaining)                                # protected (currently serving)
        self.assertIn("d", remaining)
        self.assertNotIn("a", remaining)                             # least recently used went first
        self.assertTrue(is_complete(self.store.drive_path(a)))       # still safe on Drive

    def test_no_space_and_nothing_to_evict(self):
        self.store.free_gb = lambda: 0.1
        with self.assertRaises(StoreError):
            self.store.ensure_local(entry("a", size=5.0))

    def test_state_and_progress(self):
        e = entry("a")
        self.assertEqual(self.store.state(e)["on_drive"], False)
        self.store.ensure_local(e)
        self.store.wait_for_drive_copies()
        st = self.store.state(e)
        self.assertEqual((st["on_drive"], st["local"], st["phase"]), (True, True, "ready"))


class Misc(unittest.TestCase):
    def test_provider_yaml_matches_reference_shape(self):
        models = [{"id": "m1", "name": 'Model "One"', "context_length": 32768},
                  {"id": "m2", "name": "M2", "context_length": 4096}]
        text = provider.dsh_provider_yaml("https://x.trycloudflare.com/", models)
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        doc = yaml.safe_load(text)
        p = doc["llm-pi-ai"]["providers"]["colab-hub"]
        self.assertEqual(p["baseURL"], "https://x.trycloudflare.com/v1")
        self.assertEqual((p["api"], p["apiKeyEnv"]), ("openai-completions", "COLAB_LLM_API_KEY"))
        self.assertEqual(p["compat"], {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"})
        self.assertEqual([m["contextWindow"] for m in p["models"]], [32768, 4096])
        self.assertEqual([m["maxTokens"] for m in p["models"]], [4096, 2048])
        self.assertEqual(p["models"][0]["name"], 'Model "One"')

    def test_tunnel_url_parse_and_start(self):
        self.assertEqual(tunnel.parse_quick_url("2026 INF |  https://abc-def-12.trycloudflare.com  |"),
                         "https://abc-def-12.trycloudflare.com")
        self.assertIsNone(tunnel.parse_quick_url("nothing yet"))

        class P:
            pass
        tmp = tempfile.mktemp()

        def popen(args, stdout, stderr):
            stdout.write("https://zz-top.trycloudflare.com\n")
            stdout.flush()
            self.assertEqual(args[:3], ["/cf", "tunnel", "--url"])
            return P()
        proc, url = tunnel.start_quick(8000, "/cf", tmp, popen, wait=2, sleep=lambda s: None)
        self.assertEqual(url, "https://zz-top.trycloudflare.com")

    def test_llama_fetch_commands(self):
        pinned = install.llama_fetch_commands(install.LLAMA_PIN, "/s")
        self.assertTrue(any(f"fetch -q --depth 1 {install.LLAMA_URL} {install.LLAMA_PIN}" in c for c in pinned))
        self.assertTrue(any("checkout -q FETCH_HEAD" in c for c in pinned))
        self.assertEqual(len(install.llama_fetch_commands("latest", "/s")), 1)
        self.assertIn("sm89", install.llama_tarball("/b", install.LLAMA_PIN, "89"))
        self.assertIn("4c3c5810f8", install.llama_tarball("/b", install.LLAMA_PIN, "89"))

    def test_keyfile_is_stable(self):
        path = os.path.join(tempfile.mkdtemp(), "hub", "api_key")
        k1 = keyfile.load_or_create(path)
        self.assertGreaterEqual(len(k1), 32)
        self.assertEqual(keyfile.load_or_create(path), k1)


if __name__ == "__main__":
    unittest.main()


class Resume(unittest.TestCase):
    def test_partial_download_is_kept_for_resume(self):
        tmp = tempfile.mkdtemp(prefix="resume-")
        seen = {}

        def downloader(repo, patterns, dest, token=None):
            seen["had_partial"] = (dest / "model.gguf.incomplete").exists()
            (dest / "m.gguf").write_bytes(b"x" * 10)
        store = ModelStore(tmp + "/drive", tmp + "/local", downloader=downloader, min_free_gb=0, log=lambda *_: None)
        e = entry("m")
        d = store.local_path(e)
        d.mkdir(parents=True)
        (d / "model.gguf.incomplete").write_bytes(b"half")             # no marker: an interrupted download
        store.ensure_local(e)
        self.assertTrue(seen["had_partial"])
        shutil.rmtree(tmp, ignore_errors=True)


class ConfigureDsh(unittest.TestCase):
    MODELS = [{"id": "m1", "name": "M1", "context_length": 32768}, {"id": "m2", "name": "M2", "context_length": 8192}]
    SECRET = "sk-very-secret-value-123"

    def setUp(self):
        try:
            import yaml
            self.yaml = yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        self.dir = tempfile.mkdtemp(prefix="dsh-")
        self.path = os.path.join(self.dir, "settings.yaml")

    def run_configure(self, **kw):
        from colab_llm_hub import configure_dsh
        return configure_dsh.configure("https://h.trycloudflare.com", self.SECRET, self.path,
                                       fetch=lambda u, k: self.MODELS, **kw)

    def test_merges_without_touching_other_settings(self):
        existing = {"llm-pi-ai": {"providers": {"vllm-local": {"baseURL": "http://127.0.0.1:30100/v1"}}},
                    "agent-default-model": {"provider": "vllm-local", "model": "x"}, "other": {"keep": 1}}
        open(self.path, "w").write(self.yaml.safe_dump(existing))
        self.run_configure()
        doc = self.yaml.safe_load(open(self.path))
        self.assertEqual(doc["llm-pi-ai"]["providers"]["vllm-local"], existing["llm-pi-ai"]["providers"]["vllm-local"])
        self.assertEqual(doc["agent-default-model"], existing["agent-default-model"])      # default untouched
        self.assertEqual(doc["other"], {"keep": 1})
        p = doc["llm-pi-ai"]["providers"]["colab-hub"]
        self.assertEqual((p["baseURL"], p["apiKeyEnv"]), ("https://h.trycloudflare.com/v1", "COLAB_LLM_API_KEY"))
        self.assertNotIn(self.SECRET, open(self.path).read())                               # the key is never written
        self.assertTrue(os.path.exists(self.path + ".bak"))

    def test_idempotent_and_can_set_default(self):
        self.run_configure(default="m2")
        first = open(self.path).read()
        self.run_configure(default="m2")
        self.assertEqual(open(self.path).read(), first)
        self.assertEqual(self.yaml.safe_load(first)["agent-default-model"], {"provider": "colab-hub", "model": "m2"})
        with self.assertRaises(SystemExit):
            self.run_configure(default="nope")

    def test_web_profile_patch_layout(self):
        entries = [{"id": "llm-pi-ai", "name": "@deepseek-ai/dsh-llm-pi-ai", "config": {"providers": {"a": {"x": 1}}}},
                   {"id": "agent-default-model", "name": "n", "config": {"provider": "a", "model": "z"}}]
        open(self.path, "w").write(self.yaml.safe_dump(entries))
        self.run_configure(default="m1")
        doc = self.yaml.safe_load(open(self.path))
        llm = next(e for e in doc if e["id"] == "llm-pi-ai")
        self.assertEqual(llm["config"]["providers"]["a"], {"x": 1})
        self.assertIn("colab-hub", llm["config"]["providers"])
        self.assertEqual(next(e for e in doc if e["id"] == "agent-default-model")["config"],
                         {"provider": "colab-hub", "model": "m1"})

    def test_creates_missing_file(self):
        self.run_configure()
        self.assertIn("colab-hub", self.yaml.safe_load(open(self.path))["llm-pi-ai"]["providers"])
