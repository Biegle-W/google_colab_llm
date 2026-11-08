import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from notebook_code import backends, catalog, gpu, install, keyfile, provider, tunnel
from notebook_code.store import MARKER, ModelStore, StoreError, is_complete, write_marker
from .helpers import entry, fake_downloader


class Catalog(unittest.TestCase):
    def test_shipped_catalog_is_valid(self):
        entries = catalog.load()
        self.assertGreater(len(entries), 10)
        for e in entries:
            self.assertTrue(e["files"], e["id"])
            self.assertGreater(e["size_gb"], 1, e["id"])
            # GPU-resident weights: min VRAM should cover size. Offload / experimental
            # entries (Flash-Next n-gram CPU, MiniMax-H3, NF4) may download more than fits in VRAM.
            if e.get("offload") or e.get("experimental"):
                continue
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
        t4 = catalog.visible(catalog.load(), 16.1)                            # T4: image models fit; hub LLMs mostly do not
        self.assertTrue(all(e["backend"] in catalog.IMAGE_BACKENDS for e in t4))
        self.assertIn("sd15", {e["id"] for e in t4})
        self.assertFalse(any(e["backend"] in ("llama", "vllm") for e in t4))
        self.assertTrue(any(e["id"] == "ornith-1.5-122b-q4" for e in catalog.visible(catalog.load(), 102.6)))
        self.assertFalse(any(e["id"] == "ornith-1.5-122b-q4" for e in catalog.visible(catalog.load(), 42.9)))

    def test_engine_filter(self):
        entries = catalog.load()
        llama = catalog.for_engines(entries, ["llama"])
        self.assertTrue(llama and all(e["backend"] == "llama" for e in llama))
        self.assertEqual(
            len(catalog.for_engines(entries, list(catalog.BACKENDS))),
            len(entries))
        self.assertTrue(catalog.image_entries(entries))
        self.assertTrue(all(e["backend"] in catalog.IMAGE_BACKENDS for e in catalog.image_entries(entries)))
        self.assertIn("qwen3.8-27b-w4a16-sglang", catalog.by_id(entries))
        self.assertTrue(catalog.for_engines(entries, ["strata"]))
        self.assertTrue(catalog.for_engines(entries, ["maya"]))
        self.assertTrue(catalog.for_engines(entries, ["tensorfold"]))
        self.assertTrue(catalog.for_engines(entries, ["comfyui"]))
        self.assertTrue(catalog.for_engines(entries, ["sparsa"]))
        self.assertEqual(catalog.for_engines(entries, []), [])

    def test_validation_rejects_bad_entries(self):
        good = entry("a")
        for bad in ({**good, "backend": "tgi"}, {**good, "contexts": [[30, 8192], [20, 16384]]},
                    {k: v for k, v in good.items() if k != "repo"}):
            with self.assertRaises(catalog.CatalogError):
                catalog.validate([bad])
        with self.assertRaises(catalog.CatalogError):
            catalog.validate([good, good])
        with self.assertRaises(catalog.CatalogError):
            catalog.validate([entry("d", backend="diffusers", dtype="float64")])

    def test_diffusers_context_is_max_edge(self):
        by = catalog.by_id(catalog.load())
        e = by["sdxl-base"]
        self.assertEqual(catalog.kind_for(e), "image")
        self.assertEqual(catalog.context_for(e, 24.2), 1024)
        self.assertFalse(catalog.fits(by["flux-schnell"], 24.2))
        self.assertTrue(catalog.fits(by["flux-schnell"], 42.9))
        self.assertEqual(catalog.kind_for(by["minimax-h3"]), "video")
        self.assertTrue(catalog.fits(by["flash-next-q3"], 102.6))
        self.assertFalse(catalog.fits(by["flash-next-q3"], 42.9))
        self.assertIn("glm-5.3-flash-iq1s", by)
        self.assertIn("deepseek-v4.1-flash-1017-iq1s", by)
        self.assertTrue(catalog.fits(by["glm-5.3-flash-iq1s"], 102.6))
        self.assertTrue(by["deepseek-v4.1-flash-1017-iq1s"].get("offload"))

    def test_recommend_for_gpu_bands(self):
        from notebook_code import recommend
        l4 = recommend.for_gpu(24.2)
        self.assertEqual(l4["band"], "l4")
        self.assertIn("vllm", l4["engines"])
        self.assertTrue(any(x.startswith("qwen3.8-27b") for x in l4["prefetch"]))
        self.assertNotIn("deepseek-v4.1-flash-1017-iq1s", l4["prefetch"])
        g4 = recommend.for_gpu(102.6)
        self.assertIn("flash-next-strata-iq3-s", g4["prefetch"])
        self.assertIn("glm-maya-s", g4["prefetch"])
        self.assertIn("deepseek-maya-xs", g4["prefetch"])
        self.assertIn("nemotron-3.5-bolt-74b-iq2", g4["prefetch"])
        self.assertIn("qwen3.8-27b-exl3-2bpw", g4["prefetch"])
        self.assertIn("minimax-m3.1-flash-iq2-xs-d3", g4["prefetch"])
        self.assertIn("strata", g4["engines"])
        self.assertIn("maya", g4["engines"])
        self.assertIn("tensorfold", g4["engines"])
        self.assertIn("comfyui", g4["engines"])
        self.assertIn("sparsa", g4["engines"])

    def test_recommend_for_engine(self):
        from notebook_code import recommend
        llama = recommend.for_engine("llama", 102.6)
        self.assertEqual(llama["engines"], ["llama"])
        self.assertTrue(all(not x.startswith("qwen-image") for x in llama["prefetch"]))
        self.assertIn("nemotron-3.5-bolt-74b-iq2", llama["prefetch"])
        self.assertIn("ornith-1.5-122b-q4", llama["prefetch"])
        vllm = recommend.for_engine("vllm", 24.2)
        self.assertEqual(vllm["engines"], ["vllm"])
        self.assertIn("qwen3.8-27b-w4a16", vllm["prefetch"])
        self.assertNotIn("qwen3.8-27b-q4", vllm["prefetch"])
        sgl = recommend.for_engine("sglang", 42.9)
        self.assertEqual(sgl["prefetch"], ["qwen3.8-27b-w4a16-sglang"])
        img = recommend.for_engine("diffusers", 16.1)
        self.assertIn("qwen-image-2.1-nf4", img["prefetch"])
        strata = recommend.for_engine("strata", 42.9)
        self.assertEqual(strata["engines"], ["strata"])
        self.assertIn("flash-next-strata-iq3-xxs", strata["prefetch"])
        self.assertIn("flash-next-gyro-m", strata["prefetch"])
        maya = recommend.for_engine("maya", 24.2)
        self.assertIn("glm-maya-s24", maya["prefetch"])
        tf = recommend.for_engine("tensorfold", 24.2)
        self.assertEqual(tf["prefetch"], ["qwen3.8-27b-exl3-2bpw"])
        comfy = recommend.for_engine("comfyui", 24.2)
        self.assertIn("qwen-image-2.1-turbo-comfy", comfy["prefetch"])
        sparsa = recommend.for_engine("sparsa", 102.6)
        self.assertIn("minimax-m3.1-flash-iq2-xs-d3", sparsa["prefetch"])
        with self.assertRaises(ValueError):
            recommend.for_engine("tgi", 24.2)

    def test_strata_maya_catalog_and_commands(self):
        from notebook_code import backends
        by = catalog.by_id(catalog.load())
        self.assertEqual(by["flash-next-strata-iq3-xxs"]["backend"], "strata")
        self.assertEqual(by["glm-maya-s24"]["maya_model"], "glm-maya-s24")
        cmd = backends.strata_command(by["flash-next-strata-q2"], "/tmp/missing", 32768, "strata", port=30050)
        self.assertIn("launch", cmd)
        self.assertIn("--no-tui", cmd)
        self.assertIn("q2_0", cmd)
        mcmd = backends.maya_command(by["glm-maya-s24"], "/tmp/missing", 32768, "/tmp/maya", port=30060)
        self.assertEqual(mcmd[0], "bash")
        self.assertIn("--model", mcmd)
        self.assertIn("glm-maya-s24", mcmd)

    def test_tensorfold_comfy_sparsa_commands(self):
        from notebook_code import backends
        by = catalog.by_id(catalog.load())
        self.assertEqual(by["qwen3.8-27b-exl3-2bpw"]["backend"], "tensorfold")
        self.assertEqual(by["qwen-image-2.1-turbo-comfy"]["backend"], "comfyui")
        self.assertEqual(by["minimax-m3.1-flash-iq2-xs-d3"]["backend"], "sparsa")
        self.assertIn("nemotron-3.5-bolt-74b-iq2", by)
        tf = backends.tensorfold_command(by["qwen3.8-27b-exl3-2bpw"], "/tmp/tf-model", 12288,
                                         "/bin/tensorfold-native", port=30070)
        self.assertEqual(tf[0], "/bin/tensorfold-native")
        self.assertIn("serve", tf)
        self.assertIn("--no-thinking", tf)
        self.assertEqual(tf[tf.index("--name") + 1], "qwen3.8-27b-exl3-2bpw")
        comfy = backends.comfyui_command(by["qwen-image-2.1-turbo-comfy"], "/tmp/comfy-model", 1024,
                                         "/content/ComfyUI", python="/usr/bin/python3", port=30095)
        self.assertEqual(comfy[:3], ["/usr/bin/python3", "-m", "notebook_code.comfyui_server"])
        self.assertIn("qwen-image-2.1-turbo.api.json", comfy)
        sparsa = backends.sparsa_command(by["minimax-m3.1-flash-iq2-xs-d3"], "/tmp/mm", 8192,
                                         "/tmp/Sparsa", python="/usr/bin/python3", port=30065)
        self.assertEqual(sparsa[0], "/usr/bin/python3")
        self.assertTrue(sparsa[1].endswith("sparsa.py"))
        self.assertIn("serve", sparsa)
        self.assertIn("--no-api-key", sparsa)
        gyro = backends.strata_command(by["flash-next-gyro-s"], "/tmp/missing", 32768, "strata")
        self.assertIn("gyro", gyro)
        self.assertIn("Gyro-S", gyro)


class Gpu(unittest.TestCase):
    def test_parse_smi(self):
        out = "NVIDIA L4, 23034, 8.9\nNVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0\n"
        g = gpu.parse_smi(out)
        self.assertEqual((g[0]["arch"], g[0]["vram_gb"]), ("89", 24.2))
        self.assertEqual((g[1]["arch"], g[1]["vram_gb"]), ("120", 102.6))

    def test_parse_cpuinfo_and_meminfo(self):
        cpuinfo = (
            "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n"
            "physical id\t: 0\ncore id\t\t: 0\n\n"
            "processor\t: 1\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n"
            "physical id\t: 0\ncore id\t\t: 0\n\n"
            "processor\t: 2\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n"
            "physical id\t: 0\ncore id\t\t: 1\n\n"
            "processor\t: 3\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n"
            "physical id\t: 0\ncore id\t\t: 1\n\n"
        )
        c = gpu.parse_cpuinfo(cpuinfo, logical=4)
        self.assertEqual(c["model"], "Intel(R) Xeon(R) CPU @ 2.20GHz")
        self.assertEqual((c["cores"], c["threads"], c["sockets"]), (2, 4, 1))
        mem = gpu.parse_meminfo("MemTotal:       32899012 kB\nMemAvailable:   20000000 kB\nMemFree: 1000 kB\n")
        self.assertEqual((mem["total_gb"], mem["available_gb"], mem["used_gb"]), (32.9, 20.0, 12.9))
        lines = gpu.format_host({"cpu": c, "ram": mem})
        self.assertIn("2 cores / 4 threads", lines[0])
        self.assertIn("32.9 GB total", lines[1])

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

    def test_diffusion_command(self):
        e = entry("img", backend="diffusers", pipeline="FluxPipeline", dtype="bfloat16",
                  default_steps=4, default_guidance=0.0)
        args = backends.diffusion_command(e, "/models/img", 1024, python="/usr/bin/python3", port=30090)
        self.assertEqual(args[:3], ["/usr/bin/python3", "-m", "notebook_code.diffusers_server"])
        val = lambda flag: args[args.index(flag) + 1]
        self.assertEqual((val("--model-path"), val("--model-id"), val("--max-edge"), val("--dtype")),
                         ("/models/img", "img", "1024", "bfloat16"))
        self.assertEqual((val("--pipeline"), val("--steps"), val("--guidance")),
                         ("FluxPipeline", "4", "0.0"))
        self.assertEqual(backends.kind_for(e), "image")

    def test_sglang_command(self):
        e = entry("s", backend="sglang", tool_parser="qwen3_coder", reasoning_parser="qwen3")
        args = backends.sglang_command(e, "/models/s", 32768, port=30020)
        self.assertEqual(args[1:3], ["-m", "sglang.launch_server"])
        val = lambda flag: args[args.index(flag) + 1]
        self.assertEqual((val("--model-path"), val("--served-model-name"), val("--context-length")),
                         ("/models/s", "s", "32768"))
        self.assertEqual(val("--tool-call-parser"), "qwen3_coder")
        # SGLang only accepts --default-chat-template-kwargs; --chat-template-kwargs makes launch_server exit.
        self.assertEqual(json.loads(val("--default-chat-template-kwargs")), {"enable_thinking": False})
        self.assertNotIn("--chat-template-kwargs", args)

    def test_env(self):
        self.assertEqual(backends.env_for(entry("v", backend="vllm"))["HF_HUB_OFFLINE"], "1")
        self.assertEqual(backends.env_for(entry("d", backend="diffusers"))["HF_HUB_OFFLINE"], "1")
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


class WorkspaceDrive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ws-")
        from notebook_code.workspace import Workspace
        self.ws = Workspace(self.tmp, os.path.join(self.tmp, "local")).ensure().apply_env()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_layout_settings_and_outputs(self):
        self.assertTrue(self.ws.models.is_dir())
        self.assertTrue(self.ws.builds.is_dir())
        self.assertEqual(os.environ.get("HF_HOME"), str(self.ws.hf_cache))
        self.ws.save_settings({"prefetch": ["a"], "engines": ["llama"], "port": 9000})
        loaded = self.ws.load_settings()
        self.assertEqual(loaded["prefetch"], ["a"])
        self.assertEqual(loaded["port"], 9000)
        path = self.ws.save_chat("m", [{"role": "user", "content": "hi"}], "hello")
        self.assertTrue(path.is_file())
        self.assertTrue(str(path).startswith(str(self.ws.chat)))
        img = self.ws.save_image_b64(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
            "sdxl", prompt="x")
        self.assertTrue(img.is_file())
        self.assertTrue(img.with_suffix(img.suffix + ".json").is_file())

    def test_per_engine_settings_file(self):
        from notebook_code.workspace import open_workspace
        llama = open_workspace(self.tmp, os.path.join(self.tmp, "local-l"), engine="llama")
        vllm = open_workspace(self.tmp, os.path.join(self.tmp, "local-v"), engine="vllm")
        self.assertTrue(str(llama.settings_path).endswith("settings_llama.json"))
        self.assertTrue(str(vllm.settings_path).endswith("settings_vllm.json"))
        llama.save_settings({"prefetch": ["qwen3.8-27b-q4"], "engines": ["llama"]})
        vllm.save_settings({"prefetch": ["qwen3.8-27b-w4a16"], "engines": ["vllm"]})
        self.assertEqual(llama.load_settings()["prefetch"], ["qwen3.8-27b-q4"])
        self.assertEqual(vllm.load_settings()["prefetch"], ["qwen3.8-27b-w4a16"])



class Misc(unittest.TestCase):
    def test_provider_yaml_matches_reference_shape(self):
        models = [{"id": "m1", "name": 'Model "One"', "context_length": 32768},
                  {"id": "m2", "name": "M2", "context_length": 4096},
                  {"id": "img", "name": "SDXL", "context_length": 1024, "kind": "image", "backend": "diffusers"}]
        text = provider.dsh_provider_yaml("https://x.tailnet.ts.net/", models)
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        doc = yaml.safe_load(text)
        p = doc["llm-pi-ai"]["providers"]["colab-hub"]
        self.assertEqual(p["baseURL"], "https://x.tailnet.ts.net/v1")
        self.assertEqual((p["api"], p["apiKeyEnv"]), ("openai-completions", "COLAB_LLM_API_KEY"))
        self.assertEqual(p["compat"], {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"})
        self.assertEqual([m["contextWindow"] for m in p["models"]], [32768, 4096])
        self.assertEqual([m["maxTokens"] for m in p["models"]], [4096, 2048])
        self.assertEqual(p["models"][0]["name"], 'Model "One"')
        self.assertNotIn("img", [m["id"] for m in p["models"]])

    def test_tailscale_url_parse_and_authkey_prompt(self):
        self.assertEqual(
            tunnel.parse_tailscale_url("Available on the tailnet:\nhttps://colab-vllm.tailnet.ts.net/\n"),
            "https://colab-vllm.tailnet.ts.net")
        self.assertIsNone(tunnel.parse_tailscale_url("no url yet"))
        calls = []

        def fake_getpass(prompt=""):
            calls.append(prompt)
            return "tskey-auth-TESTKEY"

        import getpass as getpass_mod
        real = getpass_mod.getpass
        getpass_mod.getpass = fake_getpass
        try:
            key = tunnel.ask_tailscale_authkey()
        finally:
            getpass_mod.getpass = real
        self.assertEqual(key, "tskey-auth-TESTKEY")
        self.assertTrue(calls)
        with self.assertRaises(ValueError):
            tunnel.start_tailscale(8000, authkey="not-a-key")

    def test_tailscale_authkey_saved_on_drive_not_settings(self):
        root = tempfile.mkdtemp()
        path = os.path.join(root, "hub", "tailscale_authkey")
        import getpass as getpass_mod
        real = getpass_mod.getpass
        getpass_mod.getpass = lambda prompt="": "tskey-auth-DRIVEKEY123456"
        try:
            k1 = tunnel.load_or_ask_tailscale_authkey(path, log=lambda *_: None)
            k2 = tunnel.load_or_ask_tailscale_authkey(path, log=lambda *_: None)  # no second prompt
        finally:
            getpass_mod.getpass = real
        self.assertEqual(k1, "tskey-auth-DRIVEKEY123456")
        self.assertEqual(k2, k1)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(open(path).read().strip(), k1)
        getpass_mod.getpass = lambda prompt="": "tskey-auth-REFRESHED999999"
        try:
            k3 = tunnel.load_or_ask_tailscale_authkey(path, refresh=True, log=lambda *_: None)
        finally:
            getpass_mod.getpass = real
        self.assertEqual(k3, "tskey-auth-REFRESHED999999")
        shutil.rmtree(root, ignore_errors=True)

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


class DownloadScope(unittest.TestCase):
    """Patterns must fetch one copy of a model, not every variant a repo carries."""

    def test_diffusers_entries_fetch_only_the_diffusers_layout(self):
        for e in catalog.load():
            if e["backend"] != "diffusers" or e.get("pipeline", "").startswith("unsupported"):
                continue
            self.assertNotIn("*", e["files"], e["id"])               # "*" pulled .ckpt/.bin/fp16/non-EMA copies
            # Repo-root single-file checkpoints (v1-5-pruned.safetensors, flux1-schnell.safetensors...) need a "/".
            weights = [p for p in e["files"] if p.endswith(".safetensors")]
            self.assertTrue(weights and all("/" in p for p in weights), e["id"])
            self.assertIn("*.fp16.safetensors", e.get("exclude", []), e["id"])

    def test_unsloth_deepseek_skips_swap_in_engram_tables(self):
        e = catalog.by_id(catalog.load())["deepseek-v4.1-flash-1017-iq1s"]
        # Each Unsloth quant already holds its Engram table as its last shard; engram/ is 610 GB of alternatives.
        self.assertFalse(any(p.startswith("engram") for p in e["files"]))

    def test_exclude_is_passed_to_the_downloader(self):
        tmp = tempfile.mkdtemp(prefix="exclude-")
        seen = {}

        def downloader(repo, patterns, dest, token=None, **kw):
            seen.update(kw, patterns=patterns)
            (dest / "unet").mkdir(parents=True, exist_ok=True)
            (dest / "unet" / "w.safetensors").write_bytes(b"x")
            (dest / (repo.split("/")[1] + ".gguf")).write_bytes(b"x")     # what a llama entry loads
        store = ModelStore(tmp + "/drive", tmp + "/local", downloader=downloader, min_free_gb=0, log=lambda *_: None)
        store.ensure_local(entry("d", backend="diffusers", files=["*/*.safetensors"], exclude=["*.fp16.safetensors"]))
        self.assertEqual(seen["exclude"], ["*.fp16.safetensors"])
        seen.clear()
        store.ensure_local(entry("l", backend="llama"))
        self.assertNotIn("exclude", seen)                             # entries without it keep the old call
        shutil.rmtree(tmp, ignore_errors=True)


class ComfyPaths(unittest.TestCase):
    def test_split_files_layout_is_registered(self):
        from notebook_code.comfyui_server import write_extra_model_paths
        root = tempfile.mkdtemp(prefix="comfy-")
        out = os.path.join(root, "extra.yaml")
        write_extra_model_paths(root, out)
        self.assertNotIn("split_files", open(out).read())
        os.makedirs(os.path.join(root, "split_files", "diffusion_models"))
        write_extra_model_paths(root, out)
        text = open(out).read()
        self.assertIn(f"base_path: {os.path.realpath(root)}/split_files", text)
        self.assertEqual(text.count("diffusion_models: diffusion_models"), 2)
        try:
            import yaml
        except ImportError:
            return
        self.assertEqual(set(yaml.safe_load(text)), {"colab_hub", "colab_hub_split_files"})
        shutil.rmtree(root, ignore_errors=True)


class SelfContainedNotebooks(unittest.TestCase):
    def test_notebooks_carry_the_current_hub_code_and_download_none(self):
        """Every notebook is self-contained: its packed hub code matches this package, and no cell clones it."""
        import base64, glob as _glob, io, re, zipfile
        from notebook_code import make_engine_notebooks as m
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        want = m.package_files()
        paths = [os.path.join(root, m.ENGINE_META[e]["notebook"]) for e in m.ENGINES]
        paths += [os.path.join(root, m.PREPARE_NOTEBOOK)] + [os.path.join(root, p) for p in m.STANDALONE_NOTEBOOKS]
        for path in paths:
            nb = json.load(open(path))
            payload = [c for c in nb["cells"] if m.PAYLOAD_TAG in c.get("metadata", {}).get("tags", [])]
            self.assertEqual(len(payload), 1, path)
            src = "".join(payload[0]["source"])
            b64 = re.search(r'HUB_ZIP = """\n(.*?)\n"""', src, re.S).group(1).replace("\n", "")
            z = zipfile.ZipFile(io.BytesIO(base64.b64decode(b64)))
            got = {n: z.read(n) for n in z.namelist()}
            self.assertEqual(got, want, f"{path} is stale: run python -m notebook_code.make_engine_notebooks")
            rest = "".join("".join(c["source"]) for c in nb["cells"] if c is not payload[0])
            self.assertNotIn("github.com/Biegle-W/google_colab_llm ", rest, path)
            self.assertNotIn("git clone -q --depth 1 https://github.com/Biegle-W", rest, path)
            self.assertNotIn("{HUB_REPO}", rest, path)

    def test_payload_cell_unpacks_a_working_package(self):
        from notebook_code import make_engine_notebooks as m
        tmp = tempfile.mkdtemp()
        try:
            src = "".join(m.payload_cell(hub_dir=tmp)["source"])
            import subprocess, sys as _sys
            code = src + "\nimport notebook_code.catalog as c; print(len(c.load()) > 10)"
            r = subprocess.run([_sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("True", r.stdout)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "notebook_code", "catalog.json")))
            self.assertFalse(os.path.isdir(os.path.join(tmp, "notebook_code", "tests")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SingleModelSelection(unittest.TestCase):
    """Each notebook downloads and serves exactly the one model picked in its MODEL dropdown."""

    def selection(self, engine, vram, model, arch=None):
        from notebook_code import make_engine_notebooks, recommend
        nb = make_engine_notebooks.build_notebook(engine)
        cfg = next("".join(c["source"]) for c in nb["cells"] if 'MODEL = "recommended"' in "".join(c["source"]))
        block = cfg[cfg.index("# The ONE model"):cfg.index("PREFETCH = [MODEL]") + len("PREFETCH = [MODEL]")]
        block = block.replace('MODEL = "recommended"', f"MODEL = {model!r}", 1)
        all_ = catalog.load()
        ns = {"catalog": catalog, "ENGINE": engine, "VRAM": vram, "RAM_GB": None, "DISK_GB": None, "ARCH": arch,
              "ENTRIES": catalog.for_engines(all_, [engine]),
              "PLAN": recommend.for_engine(engine, vram, all_, arch=arch)}
        exec(block, ns)
        return ns

    def test_dropdown_lists_only_this_engines_models(self):
        from notebook_code import make_engine_notebooks
        choices = make_engine_notebooks._model_choices("vllm")
        self.assertEqual(choices[0], "recommended")
        by = catalog.by_id(catalog.load())
        self.assertTrue(all(by[m]["backend"] == "vllm" for m in choices[1:]))

    def test_recommended_resolves_to_one_fitting_model(self):
        for engine, vram in (("llama", 24.2), ("llama", 102.6), ("vllm", 42.9), ("strata", 102.6), ("maya", 102.6)):
            ns = self.selection(engine, vram, "recommended")
            self.assertEqual(len(ns["PREFETCH"]), 1, engine)           # never a list of several large models
            self.assertTrue(catalog.fits(catalog.by_id(catalog.load())[ns["MODEL"]], vram))

    def test_explicit_pick_is_used_and_checked(self):
        self.assertEqual(self.selection("llama", 102.6, "qwen3.8-27b-q8")["PREFETCH"], ["qwen3.8-27b-q8"])
        with self.assertRaises(AssertionError):                          # does not fit an L4
            self.selection("llama", 24.2, "ornith-1.5-122b-q4")
        with self.assertRaises(AssertionError):                          # another engine's model
            self.selection("llama", 102.6, "qwen3.8-27b-fp8")

    def test_hub_serves_only_the_selected_model(self):
        from notebook_code import make_engine_notebooks
        nb = make_engine_notebooks.build_notebook("llama")
        text = "".join("".join(c["source"]) for c in nb["cells"]
                       if make_engine_notebooks.PAYLOAD_TAG not in c.get("metadata", {}).get("tags", []))
        self.assertIn('args += ["--models", MODEL]', text)
        self.assertIn('TUNNEL = saved.get("tunnel") or "tailscale"', text)        # Tailscale stays the default
        self.assertIn("tunnel.choose_tunnel(", text)                              # picked in a prompt, not code
        self.assertNotIn("npx", text)                                    # no harness runs inside the notebook

    def test_only_filter(self):
        entries = catalog.load()
        self.assertEqual([e["id"] for e in catalog.only(entries, ["qwen3.8-27b-q8"])], ["qwen3.8-27b-q8"])
        with self.assertRaises(catalog.CatalogError):
            catalog.only(entries, ["no-such-model"])


class RuntimeFit(unittest.TestCase):
    """Fit checks use VRAM, system RAM and local disk. Numbers are the user's real Colab runtimes."""
    L4 = dict(vram_gb=24.2, ram_gb=55.5, disk_gb=253.0)
    A100 = dict(vram_gb=85.9, ram_gb=175.2, disk_gb=350.0)    # /mnt/local-scratch
    G4 = dict(vram_gb=102.6, ram_gb=185.5, disk_gb=253.0)     # no scratch disk

    def why(self, mid, rt):
        return catalog.why_not(catalog.by_id(catalog.load())[mid], **rt)

    def test_offload_models_must_fit_vram_plus_ram(self):
        self.assertIn("VRAM+RAM", self.why("glm-maya-s24", self.L4))      # 96 GB pack, 80 GB of memory
        self.assertIsNone(self.why("glm-maya-s24", self.A100))
        self.assertIsNone(self.why("flash-next-strata-q2", self.L4))        # 66 GB fits 24 + 55

    def test_downloads_must_fit_the_local_disk(self):
        self.assertIn("disk", self.why("deepseek-maya-xs", self.G4))      # 282 GB on a 253 GB disk
        self.assertIsNone(self.why("deepseek-maya-xs", self.A100))         # fits the 350 GB scratch SSD
        self.assertIsNone(self.why("deepseek-v4.1-flash-1017-iq1s", self.G4))   # 233 GB + headroom < 253

    def test_streamed_engram_does_not_count_as_resident(self):
        e = catalog.by_id(catalog.load())["deepseek-v4.1-flash-1017-iq1s"]
        self.assertLess(catalog.resident_gb(e), e["size_gb"])
        self.assertTrue(catalog.fits(e, 85.9, ram_gb=60, disk_gb=300))     # 122 GB resident < 86 + 60

    def test_vram_only_check_is_unchanged_without_ram_and_disk(self):
        e = catalog.by_id(catalog.load())["glm-maya-s24"]
        self.assertTrue(catalog.fits(e, 24.2))

    def test_recommendations_respect_ram_and_disk(self):
        from notebook_code import recommend
        plan = recommend.for_engine("maya", **self.L4)
        by = catalog.by_id(catalog.load())
        self.assertTrue(all(catalog.fits(by[m], **self.L4) for m in plan["prefetch"]))
        self.assertNotIn("deepseek-maya-xs", recommend.for_engine("maya", **self.G4)["prefetch"])

    def test_local_models_dir_prefers_the_larger_disk(self):
        from notebook_code import workspace
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        free = {a: 350e9, b: 250e9}
        old = os.environ.pop("COLAB_LLM_LOCAL_MODELS", None)
        try:
            self.assertEqual(str(workspace.pick_local_models((a, b), free_bytes=free.get)), os.path.join(a, "models"))
            self.assertEqual(str(workspace.pick_local_models(("/nonexistent-disk", b), free_bytes=free.get)),
                             os.path.join(b, "models"))
            os.environ["COLAB_LLM_LOCAL_MODELS"] = "/x/models"
            self.assertEqual(str(workspace.pick_local_models((a, b), free_bytes=free.get)), "/x/models")
        finally:
            os.environ.pop("COLAB_LLM_LOCAL_MODELS", None)
            if old is not None:
                os.environ["COLAB_LLM_LOCAL_MODELS"] = old

    def test_model_disk_counts_cached_models_as_available(self):
        from notebook_code.store import model_disk_gb
        root = tempfile.mkdtemp()
        d = os.path.join(root, "m")
        os.makedirs(d)
        open(os.path.join(d, "w.gguf"), "wb").write(b"x" * 10**6)
        write_marker(d, "org/m")
        free = shutil.disk_usage(root).free / 1e9
        self.assertGreaterEqual(model_disk_gb(root), round(free, 1))


class HyperQwenAndNInfer(unittest.TestCase):
    L4, A100, G4 = (24.2, "89"), (85.9, "80"), (102.6, "120")

    def ids(self, engine, gpu):
        return [e["id"] for e in catalog.visible(catalog.for_engines(catalog.load(), [engine]), gpu[0], arch=gpu[1])]

    def test_catalog_fit_by_runtime(self):
        self.assertEqual(self.ids("hyperqwen", self.L4), [])                 # 22.5 GiB is under the 24 GB target
        self.assertEqual(self.ids("hyperqwen", self.A100), ["hyperqwen-qwen3.8-27b-batch"])   # sm80 spec fault
        self.assertEqual(len(self.ids("hyperqwen", self.G4)), 3)
        self.assertEqual(self.ids("ninfer", self.A100), [])
        self.assertEqual(self.ids("ninfer", self.L4), [])
        self.assertEqual(len(self.ids("ninfer", self.G4)), 4)
        e = catalog.by_id(catalog.load())["ninfer-qwen3.8-27b"]
        self.assertIn("sm_120", catalog.why_not(e, 85.9, arch="80"))
        self.assertIsNone(catalog.why_not(e, 85.9))                          # arch unknown -> not checked

    def test_recommended_picks(self):
        from notebook_code import recommend

        def rec(eng, gpu):
            return recommend.for_engine(eng, gpu[0], arch=gpu[1])["prefetch"]
        self.assertEqual(rec("hyperqwen", self.A100), ["hyperqwen-qwen3.8-27b-batch"])
        self.assertEqual(rec("hyperqwen", self.G4), ["hyperqwen-qwen3.8-27b-dflash2"])
        self.assertEqual(rec("ninfer", self.G4), ["ninfer-qwen3.8-27b-nvfp4"])
        self.assertEqual(rec("ninfer", self.A100), [])

    def test_hyperqwen_disk_counts_the_prepared_copy(self):
        e = catalog.by_id(catalog.load())["hyperqwen-qwen3.8-27b-mtp"]
        self.assertIsNone(catalog.why_not(e, 102.6, disk_gb=50))             # 19.5 Drive cache + 19.5 copy + 8
        self.assertIn("disk", catalog.why_not(e, 102.6, disk_gb=40))

    def test_ninfer_command_and_env(self):
        e = catalog.by_id(catalog.load())["ninfer-qwen3.8-27b-nvfp4"]
        cmd = backends.command(e, "/m/ninfer-qwen3.8-27b-nvfp4/qwen3_8_27b_nvfp4.ninfer", 240000,
                               {"ninfer": "/opt/nf/apps/ninfer-serve"})
        self.assertEqual(cmd[:2], ["/opt/nf/apps/ninfer-serve", "/m/ninfer-qwen3.8-27b-nvfp4/qwen3_8_27b_nvfp4.ninfer"])
        flags = {a: b for a, b in zip(cmd[2:], cmd[3:]) if a.startswith("--") and not b.startswith("--")}
        self.assertEqual((flags["--host"], flags["--port"], flags["--model-id"], flags["--max-context"]),
                         ("127.0.0.1", "30040", "ninfer-qwen3.8-27b-nvfp4", "240000"))
        self.assertEqual((flags["--spec"], flags["--draft-tokens"], flags["--proposal-head"]), ("mtp", "3", "optimized"))
        self.assertIn("--no-thinking", cmd)
        env = backends.env_for(e, {"ninfer": "/opt/nf/apps/ninfer-serve"})
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/opt/nf/lib"))
        from notebook_code.store import ModelStore
        tmp = tempfile.mkdtemp()
        store = ModelStore(os.path.join(tmp, "d"), os.path.join(tmp, "l"), log=lambda *_: None)
        self.assertEqual(store.model_path(e).name, "qwen3_8_27b_nvfp4.ninfer")
        shutil.rmtree(tmp, ignore_errors=True)

    def test_hyperqwen_command_and_env(self):
        e = catalog.by_id(catalog.load())["hyperqwen-qwen3.8-27b-dflash2"]
        cmd = backends.command(e, "/m/hq", 131072, {"hyperqwen": "/content/HyperQwen", "python": "py"})
        self.assertEqual(cmd[:3], ["py", "-m", "notebook_code.hyperqwen_launch"])
        flags = dict(zip(cmd[3::2], cmd[4::2]))
        self.assertEqual((flags["--mode"], flags["--spec"], flags["--port"], flags["--ctx"]),
                         ("single", "dflash2", "30030", "131072"))
        os.environ["VLLM_API_KEY"] = "x"
        try:
            self.assertNotIn("VLLM_API_KEY", backends.env_for(e))           # no key -> binds 127.0.0.1
        finally:
            del os.environ["VLLM_API_KEY"]

    def test_hyperqwen_launch_env(self):
        from notebook_code import hyperqwen_launch as hl
        tmp = tempfile.mkdtemp()
        drafter = os.path.join(tmp, "dflash2")
        os.makedirs(drafter)
        open(os.path.join(drafter, "model.safetensors"), "w").close()
        env = hl.launch_env("/p", 30030, 131072, "single", "dflash2", drafter=drafter, base={"VLLM_API_KEY": "k"})
        self.assertEqual((env["MODEL"], env["PORT"], env["MAX_LEN"], env["SPEC"], env["CTX"], env["DRAFT"]),
                         ("/p", "30030", "131072", "dflash2", "long", drafter))
        self.assertNotIn("VLLM_API_KEY", env)
        self.assertEqual(env["HOST"], "127.0.0.1")                          # loopback only; the hub does auth
        self.assertIn('{"enable_thinking":false}', env["EXTRA_ARGS"])           # reasoning off, as in the reference
        self.assertNotIn(" ", env["EXTRA_ARGS"].split("=", 1)[1])               # survives HyperQwen's word split
        self.assertNotIn("EXTRA_ARGS", hl.launch_env("/p", 1, 1, "batch", "off", base={}, thinking=True))
        self.assertEqual(hl.launch_env("/p", 1, 65536, "single", "mtp", base={})["CTX"], "fast")
        batch = hl.launch_env("/p", 1, 150000, "batch", "off", base={})
        self.assertNotIn("SPEC", batch)
        self.assertTrue(hl.script_for("/r", "batch").endswith("batch/start_qwen.sh"))
        with self.assertRaises(SystemExit):
            hl.launch_env("/p", 1, 1, "single", "dflash2", drafter=os.path.join(tmp, "none"), base={})
        shutil.rmtree(tmp, ignore_errors=True)

    def test_hyperqwen_prepare_runs_once_on_a_copy(self):
        from notebook_code import hyperqwen_launch as hl
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "models", "hq")
        os.makedirs(os.path.join(src, "dflash2"))
        open(os.path.join(src, "config.json"), "w").write("{}")
        open(os.path.join(src, ".complete.json"), "w").write('{"repo": "x"}')
        calls = []

        class R:
            returncode = 0

        def run(cmd, cwd=None):
            calls.append(cmd)
            open(os.path.join(cmd[2], "touched"), "a").write("x")          # scripts rewrite files in place
            return R()
        dest = hl.prepare("/root", src, "py", run=run, log=lambda *_: None)
        self.assertEqual(len(calls), 4)
        self.assertFalse(os.path.exists(os.path.join(src, "touched")))     # the download is never modified
        self.assertFalse(os.path.exists(os.path.join(dest, "dflash2")))
        hl.prepare("/root", src, "py", run=run, log=lambda *_: None)
        self.assertEqual(len(calls), 4)                                      # reused while the source is unchanged
        open(os.path.join(src, ".complete.json"), "w").write('{"repo": "y"}')
        hl.prepare("/root", src, "py", run=run, log=lambda *_: None)
        self.assertEqual(len(calls), 8)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_ninfer_install_refuses_other_gpus(self):
        with self.assertRaises(install.InstallError):
            install.ensure_ninfer("/tmp/x", "80", run=lambda c: "")

    def test_ninfer_dep_builds_only_when_too_old(self):
        self.assertEqual(install.ninfer_dep_commands((2, 9), (8, 5)), [])
        self.assertEqual(len(install.ninfer_dep_commands((2, 1), (8, 5))), 2)   # Ubuntu 22.04 liburing 2.1
        self.assertEqual(len(install.ninfer_dep_commands((2, 9), (7, 81))), 2)  # Ubuntu 22.04 libcurl 7.81

    def test_hyperqwen_build_applies_patches_to_the_venv(self):
        cmds = install.hyperqwen_build_commands("/content/HyperQwen", install.HYPERQWEN_PIN, python="python3")
        text = "\n".join(cmds)
        self.assertIn("vllm==0.32.0", text)
        self.assertIn("bash patches/apply.sh", text)
        self.assertIn(install.HYPERQWEN_PIN, text)

    def test_notebooks_exist_and_install_their_engine(self):
        from notebook_code import make_engine_notebooks
        for engine, call in (("hyperqwen", "install.ensure_hyperqwen("), ("ninfer", "install.ensure_ninfer(")):
            nb = make_engine_notebooks.build_notebook(engine)
            text = "".join("".join(c["source"]) for c in nb["cells"])
            self.assertIn(call, text)
            self.assertIn(f'ENGINE   = "{engine}"', text)


class PrepareOnCpu(unittest.TestCase):
    """A CPU runtime downloads models straight to Drive and builds engines for a GPU it does not have."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo_files = {"org/m": [("config.json", 2), ("model-1.safetensors", 5), ("model-2.safetensors", 7),
                                     ("onnx/model.onnx", 9), ("README.md", 3)],
                           "org/draft": [("config.json", 4), ("model.safetensors", 6)]}
        self.fetched = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lister(self, repo, token=None):
        return self.repo_files[repo]

    def fetch(self, repo, path, dest, token=None):
        self.fetched.append((repo, path))
        size = dict(self.repo_files[repo])[path]
        out = os.path.join(dest, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(b"x" * size)
        return out

    def store(self):
        return ModelStore(os.path.join(self.tmp, "drive"), os.path.join(self.tmp, "local"), log=lambda *_: None)

    def entry(self, **kw):
        e = {"id": "m", "backend": "vllm", "repo": "org/m", "files": ["*.json", "*.safetensors"],
             "exclude": ["onnx/*"], "size_gb": 0.0, "contexts": [[1, 1024]],
             "extra_downloads": [{"repo": "org/draft", "files": ["*"], "subdir": "drafter"}]}
        e.update(kw)
        return e

    def test_select_files_matches_snapshot_semantics(self):
        from notebook_code.store import select_files
        files = self.repo_files["org/m"]
        self.assertEqual([p for p, _ in select_files(files, ["*.safetensors"])],
                         ["model-1.safetensors", "model-2.safetensors"])
        self.assertEqual([p for p, _ in select_files(files, ["onnx/"])], ["onnx/model.onnx"])
        self.assertEqual(len(select_files(files, None, ["*.md"])), 4)

    def test_download_to_drive_one_file_at_a_time_then_resume(self):
        st = self.store()
        drive = st.download_to_drive(self.entry(), lister=self.lister, fetch=self.fetch)
        self.assertTrue(is_complete(drive))
        listed = json.load(open(os.path.join(drive, MARKER)))["files"]
        self.assertEqual(sorted(listed), ["config.json", "drafter/config.json", "drafter/model.safetensors",
                                          "model-1.safetensors", "model-2.safetensors"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "local", ".staging", "m")))  # staging cleaned
        os.remove(os.path.join(drive, "model-2.safetensors"))
        os.remove(os.path.join(drive, MARKER))
        self.fetched.clear()
        st.download_to_drive(self.entry(), lister=self.lister, fetch=self.fetch)
        self.assertEqual(self.fetched, [("org/m", "model-2.safetensors")])           # only the missing file
        self.fetched.clear()
        st.download_to_drive(self.entry(), lister=self.lister, fetch=self.fetch)
        self.assertEqual(self.fetched, [])                                           # complete: nothing to do
        self.assertTrue(st.state(self.entry())["on_drive"])

    def test_failed_copy_leaves_drive_incomplete(self):
        def bad_copy(src, dst, on_bytes=None):
            raise OSError("Drive upload limit")
        with self.assertRaises(OSError):
            self.store().download_to_drive(self.entry(), lister=self.lister, fetch=self.fetch, copier=bad_copy)
        self.assertFalse(is_complete(os.path.join(self.tmp, "drive", "models", "m")))

    def test_download_to_drive_refuses_engine_managed_and_checks_model_file(self):
        with self.assertRaises(StoreError):
            self.store().download_to_drive(self.entry(engine_download=True), lister=self.lister, fetch=self.fetch)
        with self.assertRaises(StoreError):
            self.store().download_to_drive(self.entry(id="m2", model_file="missing.gguf"),
                                           lister=self.lister, fetch=self.fetch)

    def test_pick_uses_the_target_not_this_runtime(self):
        from notebook_code import prepare
        self.assertEqual(prepare.pick("ninfer", "g4")["id"], "ninfer-qwen3.8-27b-nvfp4")
        self.assertEqual(prepare.pick("hyperqwen", "a100")["id"], "hyperqwen-qwen3.8-27b-batch")
        with self.assertRaises(ValueError):
            prepare.pick("ninfer", "a100")                                           # no NInfer model on sm_80
        with self.assertRaises(ValueError):
            prepare.pick("llama", "l4", "ornith-1.5-122b-q4")                        # does not fit an L4
        with self.assertRaises(ValueError):
            prepare.pick("llama", "g4", "qwen3.8-27b-fp8")                           # another engine's model
        with self.assertRaises(ValueError):
            prepare.pick("llama", "h100")

    def test_prepare_runs_download_and_build_together(self):
        from notebook_code import prepare
        started = []
        both = threading.Barrier(2, timeout=10)                  # deadlocks unless the two run concurrently

        class FakeStore:
            def download_to_drive(self, entry):
                started.append("download")
                both.wait()
                return "/drive/models/" + entry["id"]

        def builder(engine, target, builds, refresh=False, log=None):
            started.append("build")
            both.wait()
            return builds + "/cache"
        r = prepare.prepare("g4", "vllm", "recommended", drive_dir="/d", store=FakeStore(), engine_builder=builder,
                            log=lambda *_: None)
        self.assertEqual(r["errors"], {})
        self.assertEqual((r["engine_cache"], r["drive_path"]), ("/d/engines/cache", "/drive/models/" + r["model"]))

        def broken(*a, **k):
            raise install.InstallError("nvcc not found")
        r = prepare.prepare("g4", "vllm", do_download=False, engine_builder=broken, log=lambda *_: None)
        self.assertIn("nvcc not found", r["errors"]["engine"])

    def test_compiled_engines_are_left_to_the_gpu_on_small_runtimes(self):
        from notebook_code import prepare
        with self.assertRaises(prepare.Skipped):
            prepare.build_engine("ninfer", "g4", self.tmp, threads=2)
        r = prepare.prepare("g4", "llama", do_download=False, log=lambda *_: None,
                            engine_builder=None, force_compile=False) if os.cpu_count() < 8 else None
        if r is not None:
            self.assertEqual(r["errors"], {})
            self.assertIn("llama", r["skipped"]["engine"])

    def test_engine_caches_skip_the_local_install(self):
        builds = os.path.join(self.tmp, "builds")
        os.makedirs(builds)
        for tar in (install.ninfer_tarball(builds), install.hyperqwen_tarball(builds)):
            open(tar, "w").close()
        never = lambda cmd: self.fail(f"ran {cmd}")                                   # noqa: E731
        self.assertEqual(install.ensure_ninfer(builds, "120", run=never, install_now=False, log=lambda *a: None),
                         install.ninfer_tarball(builds))
        self.assertEqual(install.ensure_hyperqwen(builds, run=never, install_now=False, log=lambda *a: None),
                         install.hyperqwen_tarball(builds))
        ran = []
        wheels = install.ensure_vllm(builds, run=ran.append, install_now=False, log=lambda *a: None)
        self.assertEqual(len(ran), 1)
        self.assertIn("pip wheel", ran[0])                                          # cached, never installed here
        install.ensure_vllm(builds, run=ran.append, install_now=False, log=lambda *a: None)
        self.assertEqual(len(ran), 1)                                                # COMPLETE marker: no re-download
        self.assertTrue(os.path.isfile(os.path.join(wheels, "COMPLETE")))

    def test_runtime_targets(self):
        from notebook_code import runtimes
        self.assertEqual(runtimes.specs("G4")[3], "120")
        self.assertEqual(runtimes.specs("a100")[3], "80")

    def test_prepare_notebook(self):
        from notebook_code import make_engine_notebooks
        nb = make_engine_notebooks.build_prepare_notebook()
        self.assertNotIn("accelerator", nb["metadata"])                            # opens on a CPU runtime
        text = "".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("prepare.prepare(", text)
        self.assertIn('ENGINE   = "prepare"', text)
        self.assertNotIn("gpu.detect", text)                                         # no GPU required
        gpu_boot = "".join(make_engine_notebooks.build_notebook("vllm")["cells"][3]["source"])
        self.assertIn(gpu_boot.split("DRIVE_DIR = ")[1][:400], text)                 # same bootstrap code

    def test_cli(self):
        from notebook_code import __main__ as cli
        a = cli.parse(["prepare", "--target", "g4", "--engine", "ninfer", "--no-engine"])
        self.assertEqual((a.cmd, a.target, a.engine, a.model, a.no_engine), ("prepare", "g4", "ninfer",
                                                                             "recommended", True))


class CloudflareTunnel(unittest.TestCase):
    """Quick tunnel: URL only once it has registered with Cloudflare and the hub answers through it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.clock = [0.0]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_popen(self, lines_by_attempt):
        attempts = []

        class Proc:
            def __init__(p, cmd, stdout=None, **kw):
                attempts.append(cmd)
                p.out = stdout.name
                p.lines = list(lines_by_attempt[len(attempts) - 1])
                p.done = False

            def poll(p):
                if p.lines:
                    with open(p.out, "a") as f:
                        f.write(p.lines.pop(0) + "\n")
                return 1 if (p.done or (not p.lines and p.exits)) else None
            exits = False

            def terminate(p):
                p.done = True
        return Proc, attempts

    def sleep(self, s):
        self.clock[0] += s

    def run_quick(self, attempts_lines, probe=lambda url, key: True, **kw):
        Proc, attempts = self.fake_popen(attempts_lines)
        orig = tunnel.time.monotonic
        tunnel.time.monotonic = lambda: self.clock[0]
        try:
            proc, url = tunnel.start_cloudflare_quick(
                8000, drive_dir=self.tmp, api_key="k", binary="/bin/cloudflared", popen=Proc, sleep=self.sleep,
                dns_check=lambda host: True, probe=probe, log=lambda *a: None, **kw)
        finally:
            tunnel.time.monotonic = orig
        return proc, url, attempts

    def test_waits_for_registration_and_returns_the_url(self):
        lines = ["INF Requesting new quick Tunnel on trycloudflare.com...",
                 "INF |  https://brave-mint-otter.trycloudflare.com  |",
                 "INF Registered tunnel connection connIndex=0 location=fra"]
        _, url, attempts = self.run_quick([lines])
        self.assertEqual(url, "https://brave-mint-otter.trycloudflare.com")
        self.assertEqual(attempts[0][1:4], ["tunnel", "--url", "http://127.0.0.1:8000"])
        self.assertNotIn("--protocol", attempts[0])

    def test_retries_over_http2_when_quic_never_registers(self):
        no_url = ["ERR failed to request quick Tunnel: Post https://api.trycloudflare.com/tunnel"] + [""] * 40
        ok = ["INF |  https://calm-sea.trycloudflare.com  |", "INF Registered tunnel connection"]
        _, url, attempts = self.run_quick([no_url, ok])
        self.assertEqual(url, "https://calm-sea.trycloudflare.com")              # not api.trycloudflare.com
        self.assertEqual(attempts[1][-2:], ["--protocol", "http2"])

    def test_gives_up_with_the_log_tail(self):
        stuck = ["ERR 429 Too Many Requests"] + [""] * 40
        with self.assertRaises(RuntimeError) as cm:
            self.run_quick([stuck, stuck])
        self.assertIn("429", str(cm.exception))

    def test_returns_even_if_this_network_cannot_reach_it_yet(self):
        lines = ["INF |  https://slow-dns.trycloudflare.com  |", "INF Registered tunnel connection"]
        _, url, _ = self.run_quick([lines], probe=lambda url, key: False)
        self.assertEqual(url, "https://slow-dns.trycloudflare.com")

    def test_probe_tells_the_server_from_cloudflare_errors(self):
        import io
        import urllib.error

        def opener_for(code):
            def opener(req, timeout=None):
                if code == 200:
                    class R(io.BytesIO):
                        status = 200
                        def __enter__(s): return s
                        def __exit__(s, *a): pass
                    return R(b'{"status":"ok"}')
                raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b"{}"))
            return opener
        self.assertTrue(tunnel.probe_hub("https://x", "k", opener=opener_for(200)))
        self.assertFalse(tunnel.probe_hub("https://x", "k", opener=opener_for(530)))   # error 1033: not registered
        self.assertFalse(tunnel.probe_hub("https://x", "k", opener=opener_for(502)))
        self.assertTrue(tunnel.probe_hub("https://x", None, opener=opener_for(401)))   # the hub refusing: reached

    def test_named_tunnel_needs_token_and_hostname(self):
        with self.assertRaises(ValueError):
            tunnel.start_cloudflare_named(8000, "", "llm.example.com", binary="/bin/true")
        with self.assertRaises(ValueError):
            tunnel.start_cloudflare_named(8000, "tok", "", binary="/bin/true")

    def test_secret_saved_privately(self):
        path = os.path.join(self.tmp, "hub", "cloudflare_tunnel_token")
        import getpass
        orig = getpass.getpass
        getpass.getpass = lambda prompt="": "  secret-token  "
        try:
            self.assertEqual(tunnel.load_or_ask_secret(path, "?", log=lambda *a: None), "secret-token")
        finally:
            getpass.getpass = orig
        self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
        self.assertEqual(tunnel.load_or_ask_secret(path, "?"), "secret-token")      # reused, not re-asked

    def test_notebooks_offer_both(self):
        from notebook_code import make_engine_notebooks
        text = "".join("".join(c["source"]) for c in make_engine_notebooks.build_notebook("vllm")["cells"])
        for needle in ("tunnel.start_tailscale(", "tunnel.start_cloudflare_quick(", "tunnel.start_cloudflare_named(",
                       "tunnel.stop_cloudflare()"):
            self.assertIn(needle, text)


class TailscaleKeyFile(unittest.TestCase):
    def test_key_read_from_file_or_txt_upload(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "tailscale_authkey")
        key = "tskey-auth-k1234567890abcdef-ABCDEFGHIJKLMNOP"
        open(path, "w").close()                                                     # empty placeholder
        open(path + ".txt", "w", encoding="utf-8-sig").write(key + "\r\n")          # Windows upload with BOM
        self.assertEqual(tunnel.load_or_ask_tailscale_authkey(path, log=lambda *a: None), key)
        open(path, "w").write(key)
        os.remove(path + ".txt")
        self.assertEqual(tunnel.load_or_ask_tailscale_authkey(path, log=lambda *a: None), key)
        shutil.rmtree(tmp, ignore_errors=True)


class ReadFromDrive(unittest.TestCase):
    """VRAM + RAM models are read from Drive; SSD-streamed files are copied to local disk every run."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fetched = []
        files = {"org/m": [("config.json", 2), ("model.gguf", 50), ("engram/table-1.bin", 30),
                           ("engram/table-1.map", 3)]}

        def lister(repo, token=None):
            return files[repo]

        def fetch(repo, path, dest, token=None):
            self.fetched.append(path)
            out = os.path.join(dest, path)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "wb").write(b"x" * dict(files[repo])[path])
            return out
        self.lister, self.fetch = lister, fetch

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def store(self):
        st = ModelStore(os.path.join(self.tmp, "drive"), os.path.join(self.tmp, "local"), min_free_gb=0,
                        log=lambda *_: None)
        orig = st.download_to_drive
        st.download_to_drive = lambda e, **kw: orig(e, lister=self.lister, fetch=self.fetch, **kw)
        return st

    def entry(self, **kw):
        e = {"id": "m", "name": "M", "backend": "llama", "repo": "org/m", "files": ["*"], "size_gb": 85e-9,
             "model_file": "model.gguf", "contexts": [[1, 1024]]}
        e.update(kw)
        return e

    def test_ram_model_is_read_from_drive(self):
        st = self.store()
        path = st.resolve(self.entry())
        self.assertEqual(path, st.drive_path(self.entry()) / "model.gguf")
        self.assertFalse(st.local_path(self.entry()).exists())                  # nothing copied locally
        self.fetched.clear()
        st.resolve(self.entry())
        self.assertEqual(self.fetched, [])                                        # next run: Drive only

    def test_streamed_files_copied_rest_linked_every_run(self):
        e = self.entry(resident_gb=50e-9, stream_files=["engram/*"])           # 85 bytes, 50 resident
        self.assertTrue(catalog.streams_from_disk(e))
        st = self.store()
        path = st.resolve(e)
        local = st.local_path(e)
        self.assertEqual(path, local / "model.gguf")
        self.assertTrue((local / "model.gguf").is_symlink())                      # not streamed: from Drive
        for rel in ("engram/table-1.bin", "engram/table-1.map"):                 # streamed weights + their map
            self.assertTrue((local / rel).is_file() and not (local / rel).is_symlink())
        self.assertTrue(is_complete(local))
        shutil.rmtree(local)                                                      # a new runtime: local disk is empty
        st.resolve(e)
        self.assertTrue((local / "engram/table-1.bin").is_file())

    def test_streamed_without_patterns_copies_everything(self):
        e = self.entry(disk_stream=True)
        st = self.store()
        st.resolve(e)
        self.assertFalse(any(p.is_symlink() for p in st.local_path(e).rglob("*")))

    def test_disk_fit_counts_only_local_files(self):
        big = {"id": "x", "backend": "vllm", "size_gb": 200, "contexts": [[10, 1024]]}
        self.assertIn("local disk", catalog.why_not(big, 24, disk_gb=150))       # Drive cache lands locally
        self.assertIsNone(catalog.why_not(big, 24, disk_gb=250))
        self.assertEqual(catalog.local_disk_gb(dict(big, work_gb=20)), 220)

    def test_network_fs_detection_and_engine_flags(self):
        mounts = os.path.join(self.tmp, "mounts")
        open(mounts, "w").write("overlay / overlay rw 0 0\ndrive /content/drive fuse.drive rw 0 0\n")
        self.assertTrue(backends.on_network_fs("/content/drive/MyDrive/llm_models/models/x", mounts))
        self.assertFalse(backends.on_network_fs("/content/models/x", mounts))
        orig = backends.on_network_fs
        backends.on_network_fs = lambda p, mounts=None: str(p).startswith("/content/drive")
        try:
            e = self.entry()
            self.assertIn("--no-mmap", backends.llama_command(e, "/content/drive/m/model.gguf", 1024, "llama"))
            self.assertNotIn("--no-mmap", backends.llama_command(e, "/content/models/m/model.gguf", 1024, "llama"))
            v = backends.vllm_command(dict(e, backend="vllm"), "/content/drive/m", 1024)
            self.assertEqual(v[v.index("--safetensors-load-strategy") + 1], "eager")
        finally:
            backends.on_network_fs = orig

    def test_hyperqwen_prepared_copy_stays_off_drive(self):
        from notebook_code import hyperqwen_launch as hl
        self.assertEqual(str(hl.prepared_dir("/content/drive/MyDrive/llm_models/models/hq", "/content/models")),
                         "/content/models/.hyperqwen/hq")
        cmd = backends.command(catalog.by_id(catalog.load())["hyperqwen-qwen3.8-27b-mtp"], "/d/hq", 65536,
                               {"work_dir": "/content/models"})
        self.assertEqual(cmd[cmd.index("--work-dir") + 1], "/content/models")


class IdleGuard(unittest.TestCase):
    """The runtime is released after N idle minutes, never while anything is happening."""

    def make(self, **signals):
        from notebook_code import idle_guard
        self.t = [1000.0]
        self.cpu_total, self.cpu_idle, self.netb = [0], [0], [0]
        self.released = []
        s = {"busy": False, "last": None, "gpu": 0, "cpu_busy": 0.0, "net_rate": 0}
        s.update(signals)
        self.s = s
        self.hb = os.path.join(tempfile.mkdtemp(), "hb")

        def cpu():
            return self.cpu_total[0], self.cpu_idle[0]

        def advance(seconds):
            self.t[0] += seconds
            self.cpu_total[0] += int(100 * seconds)
            self.cpu_idle[0] += int(100 * seconds * (1 - self.s["cpu_busy"] / 2))  # 2 cores
            self.netb[0] += int(self.s["net_rate"] * seconds)
        self.advance = advance
        return idle_guard.Guard(10, interval=30, now=lambda: self.t[0], log=lambda *a: None,
                                kernels=lambda: (self.s["busy"], self.s["last"]), cpu=cpu,
                                net=lambda: self.netb[0], gpu=lambda: self.s["gpu"], heartbeat=self.hb,
                                release=lambda log: self.released.append(self.t[0]) or True, ncpu=2)

    def run_for(self, g, minutes):
        for _ in range(int(minutes * 2)):
            self.advance(30)
            if g.step():
                return True
        return False

    def test_released_after_ten_idle_minutes_not_before(self):
        g = self.make()
        self.assertFalse(self.run_for(g, 9.5))
        self.assertTrue(self.run_for(g, 1))

    def test_each_kind_of_activity_keeps_it_alive(self):
        for signal in ({"busy": True}, {"gpu": 60}, {"cpu_busy": 1.5}, {"net_rate": 5_000_000}):
            g = self.make(**signal)
            self.assertFalse(self.run_for(g, 30), signal)

    def test_recent_cell_or_api_request_restarts_the_clock(self):
        g = self.make()
        self.run_for(g, 8)
        self.s["last"] = self.t[0]                       # a cell (or a CLI exec) just ran
        self.assertFalse(self.run_for(g, 9))
        open(self.hb, "w").close()
        os.utime(self.hb, (self.t[0], self.t[0]))        # the hub answered a request
        self.assertFalse(self.run_for(g, 9))
        self.assertTrue(self.run_for(g, 2))

    def test_release_flushes_drive_first_and_waits_if_it_cannot(self):
        from notebook_code import idle_guard
        posted = []

        class R:
            status = 200
            def __enter__(s): return s
            def __exit__(s, *a): pass
        os.environ["TBE_RUNTIME_ADDR"] = "10.0.0.1:8011"
        try:
            self.assertFalse(idle_guard.release_runtime(lambda *a: None, flush=lambda log: False,
                                                        opener=lambda req, timeout=None: posted.append(req)))
            self.assertEqual(posted, [])                  # pending Drive uploads: do not release yet
            self.assertTrue(idle_guard.release_runtime(lambda *a: None, flush=lambda log: True,
                                                       opener=lambda req, timeout=None: posted.append(req) or R()))
        finally:
            del os.environ["TBE_RUNTIME_ADDR"]
        self.assertEqual((posted[0].full_url, posted[0].get_method()), ("http://10.0.0.1:8011/unassign", "POST"))

    def test_kernel_activity_parses_jupyter(self):
        from notebook_code import idle_guard
        import io

        class R(io.BytesIO):
            def __enter__(s): return s
            def __exit__(s, *a): pass
        body = json.dumps([{"execution_state": "idle", "last_activity": "2026-11-08T01:08:28.463592Z"},
                           {"execution_state": "busy", "last_activity": "2026-11-07T22:39:55Z"}]).encode()
        busy, last = idle_guard.kernel_activity(opener=lambda url, timeout=None: R(body))
        self.assertTrue(busy)
        self.assertEqual(int(last), 1794100108)
        self.assertEqual(idle_guard.kernel_activity(opener=lambda *a, **k: (_ for _ in ()).throw(OSError())),
                         (False, None))

    def test_only_serving_notebooks_start_it(self):
        from notebook_code import make_engine_notebooks
        cells = make_engine_notebooks.build_notebook("vllm")["cells"]
        serving = next(i for i, c in enumerate(cells) if "hub_proc = subprocess.Popen" in "".join(c["source"]))
        starts = [i for i, c in enumerate(cells) if "idle_guard.start(IDLE_TIMEOUT_MIN)" in "".join(c["source"])]
        self.assertEqual(starts, [serving])                  # not during bootstrap / install / download
        prep = "".join("".join(c["source"]) for c in make_engine_notebooks.build_prepare_notebook()["cells"])
        self.assertNotIn("idle_guard", prep)                 # CPU setup runtime: never timed out

    def test_not_started_on_cpu_runtimes(self):
        from notebook_code import idle_guard
        self.assertIsNone(idle_guard.start(10, accelerator=False, log=lambda *a: None))
        import glob
        orig = glob.glob
        glob.glob = lambda pattern: []
        try:
            self.assertFalse(idle_guard.has_accelerator(which=lambda n: None, env={}, exists=lambda p: False))
            self.assertTrue(idle_guard.has_accelerator(which=lambda n: None, env={"COLAB_GPU": "1"},
                                                       exists=lambda p: False))
        finally:
            glob.glob = orig
        self.assertTrue(idle_guard.has_accelerator(which=lambda n: "/usr/bin/nvidia-smi", env={}))


class TailscaleServeNotEnabled(unittest.TestCase):
    def test_enable_link_is_reported_instead_of_hanging(self):
        out = ("Serve is not enabled on your tailnet.\nTo enable, visit:\n\n"
               "         https://login.tailscale.com/f/serve?node=noYD7JjLFD21CNTRL\n")

        def run(cmd, **kw):
            raise tunnel.subprocess.TimeoutExpired(cmd, kw.get("timeout"), output=out.encode(), stderr=None)
        code, text = tunnel._serve("/ts", "/sock", ["serve", "--bg", "--yes", "http://127.0.0.1:8000"], run=run)
        self.assertIsNone(code)
        msg = tunnel.serve_error("serve", text)
        self.assertIn("https://login.tailscale.com/f/serve?node=noYD7JjLFD21CNTRL", msg)
        self.assertIn("one-time admin setting", msg)


class ApiKeyMirror(unittest.TestCase):
    def test_key_gets_a_txt_copy_phones_can_open(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "hub", "api_key")
        key = keyfile.load_or_create(path)
        self.assertEqual(open(path + ".txt").read().strip(), key)
        self.assertEqual(oct(os.stat(path + ".txt").st_mode & 0o777), "0o600")
        os.remove(path + ".txt")
        self.assertEqual(keyfile.load_or_create(path), key)                    # same key, mirror restored
        self.assertTrue(os.path.exists(path + ".txt"))
        os.remove(path)
        self.assertEqual(keyfile.load_or_create(path), key)                    # only the .txt left: reused
        shutil.rmtree(tmp, ignore_errors=True)


class PrepareMany(unittest.TestCase):
    """Batch saves: daily budget, already-saved skips, and never filling local disk."""

    def entries(self):
        mk = lambda i, gb, **kw: dict({"id": i, "name": i, "backend": "vllm", "repo": "o/" + i, "files": ["*"],
                                       "size_gb": gb, "contexts": [[1, 1024]]}, **kw)
        return [mk("a", 300), mk("b", 300), mk("c", 300), mk("done", 10), mk("eng", 5, engine_download=True)]

    class Store:
        def __init__(self):
            self.saved, self.on_drive = [], {"done"}

        def state(self, e):
            return {"on_drive": e["id"] in self.on_drive}

        def download_to_drive(self, e, staging=None):
            self.saved.append(e["id"])

    def test_budget_skips_and_defers(self):
        from notebook_code import prepare
        st = self.Store()
        r = prepare.prepare_many(["a", "b", "c", "done", "eng"], store=st, budget_gb=700, workers=1,
                                 free=lambda p: 500, log=lambda *a: None, entries=self.entries())
        self.assertEqual(st.saved, ["a", "b"])                          # 600 GB fits the 700 GB budget
        self.assertEqual((r["skipped"], r["deferred"]), (["done"], ["c"]))
        self.assertIn("eng", r["errors"])

    def test_waits_for_drive_cache_then_stops_cleanly(self):
        from notebook_code import prepare
        frees = iter([10, 10, 500, 10] + [10] * 200)                     # full, frees up, then stays full
        st = self.Store()
        r = prepare.prepare_many(["a", "b"], store=st, budget_gb=1000, workers=1, free=lambda p: next(frees),
                                 sleep=lambda s: None, wait_free_s=300, log=lambda *a: None, entries=self.entries())
        self.assertEqual(st.saved, ["a"])
        self.assertEqual(r["deferred"], ["b"])
        self.assertIn("Drive cache", r["errors"]["b"])

    def test_recommended_lists(self):
        from notebook_code import prepare
        g4 = prepare.recommended_models("g4", "chat")
        by = catalog.by_id(catalog.load())
        self.assertTrue(all(catalog.kind_for(by[m]) == "chat" for m in g4))
        self.assertTrue(all(catalog.kind_for(by[m]) != "chat" for m in prepare.recommended_models("g4", "image")))
        self.assertNotIn("deepseek-strata-ist-2.00", prepare.recommended_models("a100", "all"))


class PrepareManyResume(unittest.TestCase):
    def test_files_left_in_staging_count_as_room(self):
        from notebook_code import prepare
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "a"))
        with open(os.path.join(tmp, "a", "shard.gguf"), "wb") as f:
            f.truncate(int(90e9 / 1e3))                       # sparse stand-in, sized below
        orig = prepare.os.path.getsize
        prepare.os.path.getsize = lambda p: 90e9 if p.endswith("shard.gguf") else orig(p)

        class Store:
            saved = []
            def state(self, e): return {"on_drive": False}
            def download_to_drive(self, e, staging=None): self.saved.append(e["id"])
        entries = [{"id": "a", "name": "a", "backend": "vllm", "repo": "o/a", "files": ["*"], "size_gb": 100,
                    "contexts": [[1, 1024]]}]
        try:
            st = Store()
            r = prepare.prepare_many(["a"], store=st, staging=tmp, workers=1, free=lambda p: 60, wait_free_s=0,
                                     log=lambda *a: None, entries=entries)
        finally:
            prepare.os.path.getsize = orig
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(st.saved, ["a"])                     # 60 GB free + 90 GB already staged is enough
        self.assertEqual(r["errors"], {})


class DownloadToDriveRoom(unittest.TestCase):
    def test_single_file_model_needs_room_for_one_file_less_what_is_staged(self):
        tmp = tempfile.mkdtemp()
        st = ModelStore(os.path.join(tmp, "d"), os.path.join(tmp, "l"), log=lambda *_: None)
        e = {"id": "one", "name": "one", "backend": "llama", "repo": "o/one", "files": ["*"], "size_gb": 63.4,
             "model_file": "m.gguf", "contexts": [[1, 1024]]}
        staging = os.path.join(tmp, "stage")
        os.makedirs(staging)
        open(os.path.join(staging, "m.gguf"), "wb").close()
        fetched = []

        def fetch(repo, path, dest, token=None):
            fetched.append(path)
            p = os.path.join(dest, path)
            open(p, "wb").write(b"x" * 10)
            return p
        orig = shutil.disk_usage
        # 150 GB free: the file plus the Drive client's cached copy of it fit, so it is staged locally
        shutil.disk_usage = lambda p: type("U", (), {"free": int(150e9), "total": 0, "used": 0})()
        try:
            def sparse_copy(s, d, cb=None):                    # a full-size (sparse) file, as a real copy leaves
                os.makedirs(os.path.dirname(d), exist_ok=True)
                with open(d, "wb") as f:
                    f.truncate(int(63.4e9))
            st.download_to_drive(e, staging=staging, lister=lambda r, t=None: [("m.gguf", int(63.4e9))], fetch=fetch,
                                 copier=sparse_copy)
        except StoreError as ex:
            self.fail(f"150 GB free should be enough for one 63.4 GB file: {ex}")
        finally:
            shutil.disk_usage = orig
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(fetched, ["m.gguf"])


class UploadPacing(unittest.TestCase):
    """Copies into the Drive mount wait for the upload, so the Drive client never builds a backlog."""

    def test_waits_until_bytes_have_gone_out(self):
        from notebook_code import drive_upload
        sent = iter([0, 0, 40, 80, 100])
        logs = []
        drive_upload.wait_uploaded(100, 0, tx=lambda: next(sent) * 1_000_000, sleep=lambda s: None,
                                   now=lambda: 0, log=logs.append, label="f")
        self.assertTrue(logs)

    def test_stall_raises(self):
        from notebook_code import drive_upload
        t = [0]

        def now():
            t[0] += 60
            return t[0]
        with self.assertRaises(drive_upload.UploadStalled):
            drive_upload.wait_uploaded(10**10, 0, tx=lambda: 5, sleep=lambda s: None, now=now,
                                       stall_s=600, log=lambda *a: None)

    def test_paced_copier_copies_then_waits(self):
        from notebook_code import drive_upload
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "big")
        with open(src, "wb") as f:
            f.truncate(60_000_000)
        counter = iter([0, 0, 70_000_000])
        events = []
        copier = drive_upload.upload_paced_copier(tx=lambda: next(counter), sleep=lambda s: None,
                                                  log=lambda *a: None,
                                                  copy=lambda s, d, cb=None: events.append("copied"))
        copier(src, os.path.join(tmp, "dst"))
        self.assertEqual(events, ["copied"])
        shutil.rmtree(tmp, ignore_errors=True)

    def test_stall_stops_the_batch(self):
        from notebook_code import prepare, drive_upload

        class Store:
            calls = []
            def state(self, e): return {"on_drive": False}
            def download_to_drive(self, e, staging=None, copier=None):
                self.calls.append(e["id"])
                raise drive_upload.UploadStalled("stuck")
        mk = lambda i: {"id": i, "name": i, "backend": "vllm", "repo": "o/" + i, "files": ["*"], "size_gb": 10,
                        "contexts": [[1, 1024]]}
        st = Store()
        r = prepare.prepare_many(["a", "b"], store=st, free=lambda p: 500, log=lambda *a: None,
                                 entries=[mk("a"), mk("b")], copier=object())
        self.assertEqual(st.calls, ["a"])                       # b never started behind a stuck upload
        self.assertEqual(r["deferred"], ["b"])


class TruncatedCopyNeverMarkedComplete(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.st = ModelStore(os.path.join(self.tmp, "d"), os.path.join(self.tmp, "l"), log=lambda *_: None)
        self.e = {"id": "m", "name": "m", "backend": "vllm", "repo": "o/m", "files": ["*"], "size_gb": 0,
                  "contexts": [[1, 1024]]}
        self.listing = [("a.bin", 100), ("b.bin", 50)]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fetch(self, repo, path, dest, token=None):
        p = os.path.join(dest, path)
        open(p, "wb").write(b"x" * dict(self.listing)[path])
        return p

    def test_cut_off_copy_is_not_marked_complete(self):
        def short_copy(src, dst, cb=None):                 # what a killed process leaves behind
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(b"x" * (10 if dst.name == "a.bin" else 50))
        with self.assertRaises(StoreError):
            self.st.download_to_drive(self.e, lister=lambda r, t=None: self.listing, fetch=self.fetch, copier=short_copy)
        self.assertFalse(self.st.state(self.e)["on_drive"])

    def test_existing_truncated_model_is_detected_and_redone(self):
        from notebook_code import prepare
        drive = self.st.drive_path(self.e)
        drive.mkdir(parents=True)
        (drive / "a.bin").write_bytes(b"x" * 10)            # truncated, but recorded as complete
        (drive / "b.bin").write_bytes(b"x" * 50)
        (drive / MARKER).write_text(json.dumps({"repo": "o/m", "files": {"a.bin": 10, "b.bin": 50}}))
        self.assertTrue(self.st.state(self.e)["on_drive"])
        orig_verify = self.st.verify_against_source
        self.st.verify_against_source = lambda e: orig_verify(e, lister=lambda r, t=None: self.listing)
        orig_dl = self.st.download_to_drive
        self.st.download_to_drive = lambda e, staging=None, **kw: orig_dl(
            e, staging=staging, lister=lambda r, t=None: self.listing, fetch=self.fetch, **kw)
        r = prepare.prepare_many(["m"], store=self.st, free=lambda p: 500, log=lambda *a: None, entries=[self.e])
        self.assertEqual(r["saved"], ["m"])
        self.assertEqual((drive / "a.bin").stat().st_size, 100)
        self.assertEqual(self.st.verify_against_source(self.e), [])


class BasaltAndStrataNvfp4(unittest.TestCase):
    G4, A100 = (102.6, "120"), (85.9, "80")

    def setUp(self):
        self.by = catalog.by_id(catalog.load())

    def test_g4_only(self):
        for mid in ("basalt-flash-next-nvfp4-dq6k", "strata-nvfp4-flash-next-gptq"):
            e = self.by[mid]
            self.assertIsNone(catalog.why_not(e, 102.6, ram_gb=176, disk_gb=253, arch="120"), mid)
            self.assertIn("sm_120", catalog.why_not(e, 85.9, ram_gb=167, disk_gb=350, arch="80"))
        from notebook_code import recommend
        self.assertEqual(recommend.for_engine("basalt", 102.6, arch="120")["prefetch"], ["basalt-flash-next-nvfp4-dq6k"])
        self.assertEqual(recommend.for_engine("strata_nvfp4", 102.6, arch="120")["prefetch"],
                         ["strata-nvfp4-flash-next-gptq"])

    def test_basalt_command_and_env(self):
        e = self.by["basalt-flash-next-nvfp4-dq6k"]
        st = ModelStore("/d", "/l", log=lambda *_: None)
        path = st.model_path(e, "/d/models/basalt-flash-next-nvfp4-dq6k")
        self.assertTrue(str(path).endswith("Qwen3.8-FN-NVFP4-DQ6_K.basalt"))
        cmd = backends.command(e, path, 131072, {"basalt": "/opt/basalt"})
        self.assertEqual(cmd[:3], ["/opt/basalt/bin/basalt", "serve", str(path)])
        flags = dict(zip(cmd[3::2], cmd[4::2]))
        self.assertEqual((flags["--python"], flags["--port"], flags["--max-context"], flags["--kv"], flags["--alias"]),
                         ("/opt/basalt/venv/bin/python", "30045", "131072", "int8", e["id"]))
        self.assertNotIn("--host", cmd)                     # Basalt binds 127.0.0.1 by itself; the hub does auth
        env = backends.env_for(e, {"basalt": "/opt/basalt"})
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/opt/basalt/lib"))

    def test_strata_nvfp4_command(self):
        e = self.by["strata-nvfp4-flash-next-gptq"]
        cmd = backends.command(e, "/d/models/strata-nvfp4-flash-next-gptq", 131072,
                               {"strata_nvfp4": "/opt/snv", "python": "/py", "builds": "/d/builds"})
        self.assertEqual(cmd[:3], ["/py", "-m", "notebook_code.strata_nvfp4_launch"])
        flags = dict(zip(cmd[3::2], cmd[4::2]))
        self.assertEqual((flags["--root"], flags["--model"], flags["--family"], flags["--port"], flags["--ctx"]),
                         ("/opt/snv", "/d/models/strata-nvfp4-flash-next-gptq", "qwen-nvfp4-gptq", "30055", "131072"))
        self.assertTrue(flags["--tarball"].startswith("/d/builds/strata_nvfp4_"))
        self.assertIn("strata_nvfp4", backends.ALIAS_READY_BACKENDS)

    def test_launcher_links_and_marks(self):
        from notebook_code import strata_nvfp4_launch as L
        tmp = tempfile.mkdtemp()
        try:
            src = os.path.join(tmp, "drive", "m")
            os.makedirs(os.path.join(src, "pack"))
            open(os.path.join(src, "pack", "experts.bin"), "wb").write(b"x" * 10)
            open(os.path.join(src, "ple-fp8.gguf"), "wb").write(b"y" * 4)
            pins = {"pack/experts.bin": (10, "aa"), "ple-fp8.gguf": (4, "bb")}
            with self.assertRaises(SystemExit):              # not recorded complete by the hub yet
                L.write_marks(src, pins, log=lambda *_: None)
            open(os.path.join(src, MARKER), "w").write("{}")
            L.write_marks(src, pins, log=lambda *_: None)
            self.assertIn("sha256 aa", open(os.path.join(src, "pack", "experts.bin.done")).read())
            self.assertIn("sha256 bb", open(os.path.join(src, "ple-fp8.gguf.done")).read())
            with self.assertRaises(SystemExit):              # a size that is not the pinned revision's
                L.write_marks(src, {"ple-fp8.gguf": (5, "bb")}, log=lambda *_: None)
            link = L.link_model(os.path.join(tmp, "models"), "Owner/Repo", src)
            self.assertEqual(os.path.basename(link), "Owner--Repo")
            self.assertTrue(os.path.isfile(os.path.join(link, "pack", "experts.bin.done")))
            self.assertEqual(L.link_model(os.path.join(tmp, "models"), "Owner/Repo", src), link)    # idempotent
            args = L.setup_args("qwen-nvfp4-gptq", "/m", "/d", 30055, 131072)
            for flag in ("--yes", "--no-start", "--no-browser"):
                self.assertIn(flag, args)
            self.assertEqual(args[args.index("--context") + 1], "131072")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cuda13_upgrade_rules(self):
        def run_with(out):
            calls = []

            def run(cmd):
                calls.append(cmd)
                if "nvidia-smi" in cmd:
                    return "580.82.07"
                if "nvcc --version" in cmd:
                    return out
                return ""
            return run, calls
        for out, upgrade in (("Cuda compilation tools, release 13.0, V13.0.88", False),
                             ("Cuda compilation tools, release 13.2, V13.2.51", True),     # miscompiles sm_120
                             ("Cuda compilation tools, release 13.2, V13.2.86", False),
                             ("Cuda compilation tools, release 12.5, V12.5.82", True)):
            run, calls = run_with(out)
            install.ensure_cuda13("x", run=run, log=lambda *_: None)
            self.assertEqual(any("cuda-toolkit-13" in c for c in calls), upgrade, out)


class DirectToDriveForHugeFiles(unittest.TestCase):
    """A file larger than half the free local disk is downloaded straight into the Drive folder."""

    def test_huge_file_goes_direct_and_small_ones_are_staged(self):
        tmp = tempfile.mkdtemp()
        st = ModelStore(os.path.join(tmp, "d"), os.path.join(tmp, "l"), log=lambda *_: None)
        e = {"id": "big", "name": "big", "backend": "basalt", "repo": "o/big", "files": ["*"], "size_gb": 131.7,
             "model_file": "m.basalt", "contexts": [[1, 1024]]}
        listing = [("m.basalt", 120), ("README.md", 5)]
        where, waited = {}, []

        def fetch(repo, path, dest, token=None):
            where[path] = str(dest)
            p = os.path.join(dest, path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(b"x" * dict(listing)[path])
            return p

        def copier(s, d, cb=None):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copyfile(s, d)
        copier.tx = lambda: 0
        copier.wait_uploaded = lambda size, start, label="": waited.append((size, label))
        orig = shutil.disk_usage
        shutil.disk_usage = lambda p: type("U", (), {"free": int(20e9 + 200), "total": 0, "used": 0})()
        try:
            st.download_to_drive(e, staging=os.path.join(tmp, "stage"), lister=lambda r, t=None: listing,
                                 fetch=fetch, copier=copier)
        finally:
            shutil.disk_usage = orig
        drive = str(st.drive_path(e))
        self.assertEqual(where["m.basalt"], drive)                       # 120 > (200) / 2: straight to Drive
        self.assertEqual(where["README.md"], os.path.join(tmp, "stage"))  # 5: staged as usual
        self.assertEqual(waited, [(120, "m.basalt")])
        self.assertTrue(is_complete(st.drive_path(e)))
        shutil.rmtree(tmp, ignore_errors=True)


class TunnelPrompt(unittest.TestCase):
    """The tunnel is picked from a prompt in the cell output; nothing in the notebook code is edited."""

    def pick(self, answers, **kw):
        it = iter(answers)
        return tunnel.choose_tunnel(ask=lambda prompt: next(it), log=lambda *a: None, **kw)

    def test_enter_keeps_the_last_choice_and_numbers_or_names_pick(self):
        self.assertEqual(self.pick([""]), ("tailscale", False))
        self.assertEqual(self.pick([""], last="cloudflare"), ("cloudflare", False))
        self.assertEqual(self.pick(["2"]), ("cloudflare", False))
        self.assertEqual(self.pick(["cloudflare-named"]), ("cloudflare-named", False))
        self.assertEqual(self.pick(["9", "x", "3"]), ("cloudflare-named", False))       # asks again
        self.assertEqual(self.pick(["1 new"]), ("tailscale", True))
        self.assertEqual(self.pick(["new"], last="cloudflare-named"), ("cloudflare-named", True))

    def test_limited_choices_and_remembered_file(self):
        tmp = tempfile.mkdtemp()
        try:
            f = os.path.join(tmp, "hub", "tunnel_t4")
            self.assertEqual(self.pick(["3", "2"], choices=("tailscale", "cloudflare"), remember=f),
                             ("cloudflare", False))
            self.assertEqual(open(f).read(), "cloudflare")
            self.assertEqual(self.pick([""], choices=("tailscale", "cloudflare"), remember=f), ("cloudflare", False))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_hostname_prompt(self):
        it = iter(["", "https://llm.example.com/"])
        self.assertEqual(tunnel.ask_hostname(ask=lambda p: next(it), log=lambda *a: None), "llm.example.com")
        self.assertEqual(tunnel.ask_hostname("a.b.com", ask=lambda p: "", log=lambda *a: None), "a.b.com")

    def test_engine_notebooks_have_no_tunnel_form_field(self):
        import glob as _glob
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for nb in _glob.glob(os.path.join(root, "notebooks", "*", "*.ipynb")):
            text = open(nb, encoding="utf-8").read()
            self.assertNotIn('TUNNEL = \\"tailscale\\"  # @param', text, nb)
            self.assertNotIn('TUNNEL      = \\"tailscale\\"  # @param', text, nb)
            if "start_tailscale" in text:
                self.assertIn("choose_tunnel", text, nb)


class SecretsOpenOnAPhone(unittest.TestCase):
    """Tailscale key and Cloudflare token are kept as .txt files too (Drive's phone apps open those)."""

    def test_tailscale_key_txt_preferred_and_mirrored(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "hub", "tailscale_authkey")
            os.makedirs(os.path.dirname(p))
            open(p, "w").write("tskey-auth-old-0123456789\n")
            self.assertEqual(tunnel.load_or_ask_tailscale_authkey(p, log=lambda *a: None), "tskey-auth-old-0123456789")
            self.assertEqual(open(p + ".txt").read().strip(), "tskey-auth-old-0123456789")     # mirrored
            open(p + ".txt", "w").write("tskey-auth-new-9876543210\n")                        # edited on a phone
            self.assertEqual(tunnel.load_or_ask_tailscale_authkey(p, log=lambda *a: None), "tskey-auth-new-9876543210")
            self.assertEqual(open(p).read().strip(), "tskey-auth-new-9876543210")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cloudflare_token_written_as_txt_too(self):
        import getpass as gp
        tmp = tempfile.mkdtemp()
        orig = gp.getpass
        gp.getpass = lambda prompt="": "cf-token-abcdef"
        try:
            p = os.path.join(tmp, "hub", "cloudflare_tunnel_token")
            self.assertEqual(tunnel.load_or_ask_secret(p, "token: ", log=lambda *a: None), "cf-token-abcdef")
            self.assertEqual(open(p + ".txt").read().strip(), "cf-token-abcdef")
            gp.getpass = lambda prompt="": self.fail("should not ask again")
            self.assertEqual(tunnel.load_or_ask_secret(p, "token: ", log=lambda *a: None), "cf-token-abcdef")
        finally:
            gp.getpass = orig
            shutil.rmtree(tmp, ignore_errors=True)


class ColabSecretsFirst(unittest.TestCase):
    def test_tailscale_secret_wins_and_is_backed_up_on_drive(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "hub", "tailscale_authkey")
            os.makedirs(os.path.dirname(p))
            open(p + ".txt", "w").write("tskey-auth-drive-0123456789\n")
            secrets = {"TAILSCALE_AUTHKEY": "tskey-auth-secret-9876543210"}
            self.assertEqual(tunnel.load_or_ask_tailscale_authkey(p, log=lambda *a: None, secret=secrets.get),
                             "tskey-auth-secret-9876543210")
            self.assertEqual(open(p + ".txt").read().strip(), "tskey-auth-secret-9876543210")
            # no secret (or not a tskey): Drive is used, nothing is asked
            self.assertEqual(tunnel.load_or_ask_tailscale_authkey(p, log=lambda *a: None, secret={}.get),
                             "tskey-auth-secret-9876543210")

            def refused(name):
                raise RuntimeError("notebook access not granted")
            self.assertEqual(tunnel.load_or_ask_tailscale_authkey(p, log=lambda *a: None, secret=refused),
                             "tskey-auth-secret-9876543210")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cloudflare_token_secret(self):
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "hub", "cloudflare_tunnel_token")
            v = tunnel.load_or_ask_secret(p, "token: ", env_name="CLOUDFLARE_TUNNEL_TOKEN", log=lambda *a: None,
                                          secret={"CLOUDFLARE_TUNNEL_TOKEN": "cf-secret"}.get)
            self.assertEqual(v, "cf-secret")
            self.assertEqual(open(p + ".txt").read().strip(), "cf-secret")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class DriveLayoutMigration(unittest.TestCase):
    def tree(self, root):
        return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs)

    def make_old(self, base):
        old = os.path.join(base, "llm_models")
        for rel in ("models/m1/a.bin", "models/hf_hub/x", "builds/tools/tailscale", "config/settings_llama.json",
                    "hub/api_key.txt", "hf_cache/token", "logs/hub_logs/e.log", "outputs/chat/c.json"):
            os.makedirs(os.path.dirname(os.path.join(old, rel)), exist_ok=True)
            open(os.path.join(old, rel), "w").write(rel)
        return old

    def test_old_layout_is_renamed(self):
        from notebook_code import workspace as w
        base = tempfile.mkdtemp()
        try:
            self.make_old(base)
            new = os.path.join(base, "colab_model_server")
            w.migrate_drive_layout(new, log=lambda *a: None)
            self.assertFalse(os.path.exists(os.path.join(base, "llm_models")))
            self.assertEqual(self.tree(new), sorted([
                "models/m1/a.bin", "models/unsloth_studio/x", "engines/tools/tailscale",
                "settings/settings_llama.json", "access/api_key.txt", "huggingface/token",
                "logs/engines/e.log", "outputs/chat/c.json"]))
            w.migrate_drive_layout(new, log=lambda *a: None)          # idempotent
            ws = w.Workspace(new, local_models=os.path.join(base, "local"), engine="llama")
            self.assertEqual(str(ws.settings_path), os.path.join(new, "settings", "settings_llama.json"))
            self.assertEqual(str(ws.hub), os.path.join(new, "access"))
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_merges_into_a_new_root_another_notebook_made(self):
        from notebook_code import workspace as w
        base = tempfile.mkdtemp()
        try:
            self.make_old(base)
            new = os.path.join(base, "colab_model_server")
            os.makedirs(os.path.join(new, "huggingface"))              # e.g. the Unsloth notebook ran first
            open(os.path.join(new, "huggingface", "new"), "w").write("n")
            w.migrate_drive_layout(new, log=lambda *a: None)
            t = self.tree(new)
            for rel in ("models/m1/a.bin", "engines/tools/tailscale", "access/api_key.txt", "huggingface/token",
                        "huggingface/new"):
                self.assertIn(rel, t)
            self.assertFalse(os.path.exists(os.path.join(base, "llm_models")))
        finally:
            shutil.rmtree(base, ignore_errors=True)
