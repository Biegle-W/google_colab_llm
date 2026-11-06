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
        from colab_llm_hub import recommend
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
        from colab_llm_hub import recommend
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
        from colab_llm_hub import backends
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
        from colab_llm_hub import backends
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
        self.assertEqual(comfy[:3], ["/usr/bin/python3", "-m", "colab_llm_hub.comfyui_server"])
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
        self.assertEqual(args[:3], ["/usr/bin/python3", "-m", "colab_llm_hub.diffusers_server"])
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
        from colab_llm_hub.workspace import Workspace
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
        from colab_llm_hub.workspace import open_workspace
        llama = open_workspace(self.tmp, os.path.join(self.tmp, "local-l"), engine="llama")
        vllm = open_workspace(self.tmp, os.path.join(self.tmp, "local-v"), engine="vllm")
        self.assertTrue(str(llama.settings_path).endswith("settings_llama.json"))
        self.assertTrue(str(vllm.settings_path).endswith("settings_vllm.json"))
        llama.save_settings({"prefetch": ["qwen3.8-27b-q4"], "engines": ["llama"]})
        vllm.save_settings({"prefetch": ["qwen3.8-27b-w4a16"], "engines": ["vllm"]})
        self.assertEqual(llama.load_settings()["prefetch"], ["qwen3.8-27b-q4"])
        self.assertEqual(vllm.load_settings()["prefetch"], ["qwen3.8-27b-w4a16"])

    def test_hub_package_cache_roundtrip(self):
        from colab_llm_hub import install
        src = tempfile.mkdtemp(prefix="hubsrc-")
        pkg = os.path.join(src, "colab_llm_hub")
        os.makedirs(pkg)
        open(os.path.join(pkg, "x.py"), "w").write("v=1\n")
        install.cache_hub_package(src, str(self.ws.builds), ref="branch/x", log=lambda *_: None)
        dest = tempfile.mkdtemp(prefix="hubdst-")
        self.assertTrue(install.restore_hub_package(str(self.ws.builds), dest, ref="branch/x", log=lambda *_: None))
        self.assertTrue(os.path.isfile(os.path.join(dest, "colab_llm_hub", "x.py")))
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)


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
        return configure_dsh.configure("https://h.tailnet.ts.net", self.SECRET, self.path,
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
        self.assertEqual((p["baseURL"], p["apiKeyEnv"]), ("https://h.tailnet.ts.net/v1", "COLAB_LLM_API_KEY"))
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
        from colab_llm_hub.comfyui_server import write_extra_model_paths
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


class HubPackageCache(unittest.TestCase):
    def test_interrupted_refresh_is_not_marked_complete(self):
        tmp = tempfile.mkdtemp(prefix="hubpkg-")
        src = os.path.join(tmp, "src")
        os.makedirs(os.path.join(src, "colab_llm_hub"))
        open(os.path.join(src, "colab_llm_hub", "__init__.py"), "w").write("")
        builds = os.path.join(tmp, "builds")
        install.cache_hub_package(src, builds, ref="main", log=lambda *a: None)
        marker = os.path.join(builds, "hub_pkg", "main", "COMPLETE")
        self.assertTrue(os.path.exists(marker))
        orig = install.shutil.copytree
        install.shutil.copytree = lambda *a, **k: (_ for _ in ()).throw(OSError("Drive went away"))
        try:
            with self.assertRaises(OSError):
                install.cache_hub_package(src, builds, ref="main", log=lambda *a: None)
        finally:
            install.shutil.copytree = orig
        self.assertFalse(os.path.exists(marker))                      # the half-copied cache is not trusted
        self.assertFalse(install.restore_hub_package(builds, os.path.join(tmp, "out"), ref="main", log=lambda *a: None))
        shutil.rmtree(tmp, ignore_errors=True)

    def test_notebooks_fetch_from_github_before_using_the_drive_copy(self):
        nb = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "llama",
                                         "llama_gpu.ipynb")))
        boot = "".join(nb["cells"][2]["source"])
        self.assertLess(boot.index("git clone"), boot.index("shutil.copytree(cached"))
        self.assertIn("if fresh:", boot)                              # the Drive copy is refreshed after a fetch


class SingleModelSelection(unittest.TestCase):
    """Each notebook downloads and serves exactly the one model picked in its MODEL dropdown."""

    def selection(self, engine, vram, model, arch=None):
        from colab_llm_hub import make_engine_notebooks, recommend
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
        from colab_llm_hub import make_engine_notebooks
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
        from colab_llm_hub import make_engine_notebooks
        nb = make_engine_notebooks.build_notebook("llama")
        text = "".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn('args += ["--models", MODEL]', text)
        self.assertNotIn("cloudflare", text.lower())
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
        from colab_llm_hub import recommend
        plan = recommend.for_engine("maya", **self.L4)
        by = catalog.by_id(catalog.load())
        self.assertTrue(all(catalog.fits(by[m], **self.L4) for m in plan["prefetch"]))
        self.assertNotIn("deepseek-maya-xs", recommend.for_engine("maya", **self.G4)["prefetch"])

    def test_local_models_dir_prefers_the_larger_disk(self):
        from colab_llm_hub import workspace
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
        from colab_llm_hub.store import model_disk_gb
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
        from colab_llm_hub import recommend

        def rec(eng, gpu):
            return recommend.for_engine(eng, gpu[0], arch=gpu[1])["prefetch"]
        self.assertEqual(rec("hyperqwen", self.A100), ["hyperqwen-qwen3.8-27b-batch"])
        self.assertEqual(rec("hyperqwen", self.G4), ["hyperqwen-qwen3.8-27b-dflash2"])
        self.assertEqual(rec("ninfer", self.G4), ["ninfer-qwen3.8-27b-nvfp4"])
        self.assertEqual(rec("ninfer", self.A100), [])

    def test_hyperqwen_disk_counts_the_prepared_copy(self):
        e = catalog.by_id(catalog.load())["hyperqwen-qwen3.8-27b-mtp"]
        self.assertIsNone(catalog.why_not(e, 102.6, disk_gb=60))
        self.assertIn("disk", catalog.why_not(e, 102.6, disk_gb=40))         # 19.5 download + 19.5 copy + 8

    def test_ninfer_command_and_env(self):
        e = catalog.by_id(catalog.load())["ninfer-qwen3.8-27b-nvfp4"]
        cmd = backends.command(e, "/m/ninfer-qwen3.8-27b-nvfp4/qwen3_8_27b_nvfp4.ninfer", 240000,
                               {"ninfer": "/opt/nf/apps/ninfer-serve"})
        self.assertEqual(cmd[:2], ["/opt/nf/apps/ninfer-serve", "/m/ninfer-qwen3.8-27b-nvfp4/qwen3_8_27b_nvfp4.ninfer"])
        flags = dict(zip(cmd[2::2], cmd[3::2]))
        self.assertEqual((flags["--host"], flags["--port"], flags["--model-id"], flags["--max-context"]),
                         ("127.0.0.1", "30040", "ninfer-qwen3.8-27b-nvfp4", "240000"))
        self.assertEqual((flags["--spec"], flags["--draft-tokens"], flags["--proposal-head"]), ("mtp", "3", "optimized"))
        env = backends.env_for(e, {"ninfer": "/opt/nf/apps/ninfer-serve"})
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/opt/nf/lib"))
        from colab_llm_hub.store import ModelStore
        tmp = tempfile.mkdtemp()
        store = ModelStore(os.path.join(tmp, "d"), os.path.join(tmp, "l"), log=lambda *_: None)
        self.assertEqual(store.model_path(e).name, "qwen3_8_27b_nvfp4.ninfer")
        shutil.rmtree(tmp, ignore_errors=True)

    def test_hyperqwen_command_and_env(self):
        e = catalog.by_id(catalog.load())["hyperqwen-qwen3.8-27b-dflash2"]
        cmd = backends.command(e, "/m/hq", 131072, {"hyperqwen": "/content/HyperQwen", "python": "py"})
        self.assertEqual(cmd[:3], ["py", "-m", "colab_llm_hub.hyperqwen_launch"])
        flags = dict(zip(cmd[3::2], cmd[4::2]))
        self.assertEqual((flags["--mode"], flags["--spec"], flags["--port"], flags["--ctx"]),
                         ("single", "dflash2", "30030", "131072"))
        os.environ["VLLM_API_KEY"] = "x"
        try:
            self.assertNotIn("VLLM_API_KEY", backends.env_for(e))           # no key -> binds 127.0.0.1
        finally:
            del os.environ["VLLM_API_KEY"]

    def test_hyperqwen_launch_env(self):
        from colab_llm_hub import hyperqwen_launch as hl
        tmp = tempfile.mkdtemp()
        drafter = os.path.join(tmp, "dflash2")
        os.makedirs(drafter)
        open(os.path.join(drafter, "model.safetensors"), "w").close()
        env = hl.launch_env("/p", 30030, 131072, "single", "dflash2", drafter=drafter, base={"VLLM_API_KEY": "k"})
        self.assertEqual((env["MODEL"], env["PORT"], env["MAX_LEN"], env["SPEC"], env["CTX"], env["DRAFT"]),
                         ("/p", "30030", "131072", "dflash2", "long", drafter))
        self.assertNotIn("VLLM_API_KEY", env)
        self.assertEqual(env["HOST"], "127.0.0.1")                          # loopback only; the hub does auth
        self.assertEqual(hl.launch_env("/p", 1, 65536, "single", "mtp", base={})["CTX"], "fast")
        batch = hl.launch_env("/p", 1, 150000, "batch", "off", base={})
        self.assertNotIn("SPEC", batch)
        self.assertTrue(hl.script_for("/r", "batch").endswith("batch/start_qwen.sh"))
        with self.assertRaises(SystemExit):
            hl.launch_env("/p", 1, 1, "single", "dflash2", drafter=os.path.join(tmp, "none"), base={})
        shutil.rmtree(tmp, ignore_errors=True)

    def test_hyperqwen_prepare_runs_once_on_a_copy(self):
        from colab_llm_hub import hyperqwen_launch as hl
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
        from colab_llm_hub import make_engine_notebooks
        for engine, call in (("hyperqwen", "install.ensure_hyperqwen("), ("ninfer", "install.ensure_ninfer(")):
            nb = make_engine_notebooks.build_notebook(engine)
            text = "".join("".join(c["source"]) for c in nb["cells"])
            self.assertIn(call, text)
            self.assertIn(f'ENGINE   = "{engine}"', text)
