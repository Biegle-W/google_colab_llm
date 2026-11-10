"""Generate the Colab notebooks: notebooks/<runtime>/<engine>/<model>.ipynb, one per recommended model.

Run from the repo root:
  python3 -m notebook_code.make_engine_notebooks
"""
from __future__ import annotations

import json
from pathlib import Path

from . import catalog
from .recommend import ENGINE_META, ENGINES, INDEX_NOTEBOOK, RUNTIMES, model_notebook, notebook_plan

REPO = "https://github.com/Biegle-W/google_colab_llm"
PREPARE_NOTEBOOK = "notebooks/cpu/prepare_cpu.ipynb"
BRANCH = "main"
ROOT = Path(__file__).resolve().parents[1]


def _src(lines):
    out = []
    for line in lines:
        out.append(line if line.endswith("\n") else line + "\n")
    return out


PAYLOAD_TAG = "notebook-code"
PACKAGE_DIR = Path(__file__).resolve().parent
_SKIP_DIRS = {"tests", "__pycache__"}
_SKIP_FILES = {"make_engine_notebooks.py"}


def package_files(pkg=PACKAGE_DIR):
    """{relative path: bytes} of the hub package as it runs on Colab (no tests, docs or generator)."""
    out = {}
    for path in sorted(Path(pkg).rglob("*")):
        rel = path.relative_to(pkg)
        if path.is_dir() or _SKIP_DIRS & set(rel.parts) or rel.name in _SKIP_FILES or path.suffix in (".pyc", ".md"):
            continue
        out[str(Path("notebook_code") / rel)] = path.read_bytes()
    return out


def package_zip(files=None):
    """The package as a deterministic zip (fixed timestamps, sorted), so unchanged code gives an unchanged notebook."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted((files or package_files()).items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return buf.getvalue()


def payload_cell(hub_dir="/content/notebook_code"):
    """A collapsed code cell holding the whole hub package; running it unpacks it on the runtime.

    The notebook needs nothing from GitHub: this cell is the code the later cells run."""
    import base64
    import hashlib
    data = package_zip()
    b64 = base64.b64encode(data).decode()
    digest = hashlib.sha256(data).hexdigest()[:12]
    lines = ["# @title Hub code, packed in this notebook (run this cell; nothing is downloaded)",
             "# Generated from the notebook_code package; unpacked to " + hub_dir + " on this runtime.",
             "import base64, io, os, shutil, sys, zipfile",
             f'HUB_DIR = "{hub_dir}"',
             f'HUB_CODE_VERSION = "{digest}"',
             'HUB_ZIP = """']
    lines += [b64[i:i + 100] for i in range(0, len(b64), 100)]
    lines += ['"""',
              "shutil.rmtree(HUB_DIR, ignore_errors=True)",
              "zipfile.ZipFile(io.BytesIO(base64.b64decode(HUB_ZIP))).extractall(HUB_DIR)",
              "del HUB_ZIP",
              "if HUB_DIR not in sys.path:",
              "    sys.path.insert(0, HUB_DIR)",
              "for _m in [m for m in sys.modules if m == \"notebook_code\" or m.startswith(\"notebook_code.\")]:",
              "    del sys.modules[_m]          # a re-run picks up this cell's code, not an older import",
              "print(\"Hub code\", HUB_CODE_VERSION, \"unpacked to\", HUB_DIR)"]
    cell = _code(lines)
    cell["metadata"] = {"cellView": "form", "tags": [PAYLOAD_TAG]}
    return cell


def _md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def _code(lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(lines)}


def _badge(notebook):
    href = (
        f"https://colab.research.google.com/github/Biegle-W/google_colab_llm/blob/"
        f"{BRANCH.replace('/', '%2F')}/{notebook}"
    )
    return (
        f'<a href="{href}" target="_parent">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )


SIBLING_NOTEBOOKS = "other runtimes, engines and models: `notebooks/README.md`"


def _model_table(engine):
    if engine == "llama":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `qwen3.8-27b-q4` / `q8` / `bf16` | Qwen3.8-27B GGUF |",
            "| `nemotron-3.5-bolt-74b-iq2` | Native IQ2 Bolt (workstation selected) |",
            "| `sage-a1.5-q4`, `ornith-1.5-122b-q4` | Sage / Ornith on A100–G4 |",
            "| `flash-next-iq1s` / `flash-next-q3` | Flash-Next fallback (prefer **Strata**) |",
            "| `glm-5.3-flash-iq1s` / `iq3` | GLM fallback (prefer **Maya**) |",
            "| `deepseek-v4.1-flash-1017-iq1s` | DeepSeek fallback (prefer **Maya** / **Strata**) |",
        ]
    if engine == "vllm":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `qwen3.8-27b-w4a16` | HyperQwen-compatible AutoRound (L4+) |",
            "| `qwen3.8-27b-fp8` / `bf16-vllm` | Dense FP8 / BF16 |",
            "| `sage-a1.5-fp8`, `nemotron-3.5-bolt-74b-w4a16` | Sage + experimental Bolt |",
            "| `qwen3-14b-fp8`, `qwen3.6-35b-a3b-fp8` | Other vLLM catalog entries |",
        ]
    if engine == "sglang":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `qwen3.8-27b-w4a16-sglang` | Same W4A16 weights as vLLM, SGLang runtime |",
            "",
            "Flash-Next / GLM / DeepSeek official SGLang recipes are multi-GPU — use **Strata** / **Maya**.",
        ]
    if engine == "strata":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `flash-next-strata-q2` / `iq3-xxs` / `iq3-s` | ISTA Flash-Next packs |",
            "| `flash-next-gyro-s` / `gyro-m` | agentionai Gyro (smallest memory) |",
            "| `flash-next-swift-iq2-xs` / `iq3-xxs` | UkisAI Swift 1.5 fine-tune |",
            "| `qwen36-strata-q4` / `q6` | Qwen3.6-35B-A3B via `--family qwen36` |",
            "| `deepseek-strata-ist-2.00` | DeepSeek via `strata-dsv41` (~257 GB) |",
            "",
            "Official: [Niko1221/Strata](https://github.com/Niko1221/Strata) `strata launch --family … --yes --no-tui`.",
        ]
    if engine == "maya":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `glm-maya-s24` | GLM-5.3-Flash for ≤24 GB VRAM (+14% decode) |",
            "| `glm-maya-s` / `glm-maya-xs` | Default / smallest Maya GLM packs |",
            "| `deepseek-maya-xs` / `deepseek-maya-s` | DeepSeek via `maya-ds4` (~281 / ~348 GB) |",
            "",
            "Official: [mw00/project-maya](https://github.com/mw00/project-maya) `./setup.sh --model glm-maya-s24 --yes`.",
        ]
    if engine == "tensorfold":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `qwen3.8-27b-exl3-2bpw` | Mia EXL3 2.0bpw + DFlash2 4-bit (12K default) |",
            "",
            "Official: [ashhart/TensorFold](https://github.com/ashhart/TensorFold) "
            "`tensorfold-native serve … --drafter … --drafter-bits 4`.",
        ]
    if engine == "comfyui":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `qwen-image-2.1-turbo-comfy` | Qwen-Image 2.1 Turbo INT8 + INT8 encoder (L4, 8 steps) |",
            "| `qwen-image-2.1-turbo-bf16-comfy` | Turbo, full precision (A100 / G4, 8 steps) |",
            "| `qwen-image-2.1-bf16-comfy` | Base model, full precision (A100 / G4, 25 steps, best images) |",
            "| `flux3-klein-comfy` | FLUX.3 Klein |",
            "| `minimax-h3-l4-comfy` | MiniMax H3 video + audio: pruned INT8 (L4, slow, short 480p clips) |",
            "| `minimax-h3-comfy` | MiniMax H3: INT8 + INT8 encoder (A100) |",
            "| `minimax-h3-g4-comfy` | MiniMax H3: INT8 + NVFP4 encoder (G4) |",
            "| `wan22-5b-comfy` | Wan 2.2 5B (prefetch; UI workflow only) |",
            "",
            "Video: `POST /v1/videos/generations` with `prompt`, `size` (e.g. `832x480`) and `seconds` "
            "(default 5); the MP4 comes back as `b64_mp4`.",
            "",
            "UI graphs under `notebook_code/workflows/*.ui.json` (custom NvidiaResident nodes are optional).",
        ]
    if engine == "sparsa":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `minimax-m3.1-flash-iq2-xs-d3` | Default Sparsa 2-bit pick (~146 GB) |",
            "| `minimax-m3.1-flash-iq2-kt` | Sparsa-only IQ2_KT (~124 GB) |",
            "",
            "Official: [LyJonathan/Sparsa](https://github.com/LyJonathan/Sparsa) "
            "`python sparsa.py serve --model … --no-api-key`.",
        ]
    if engine == "hyperqwen":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `hyperqwen-qwen3.8-27b-dflash2` | Single-user, DFlash2 block drafter (fastest; not A100) |",
            "| `hyperqwen-qwen3.8-27b-mtp` | Single-user, MTP speculation (not A100) |",
            "| `hyperqwen-qwen3.8-27b-batch` | Batch / throughput, 64 concurrent, no speculation (A100 OK) |",
            "",
            "Official: [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) patched vLLM 0.32.0, "
            "`single-user/start_qwen.sh` / `batch/start_qwen.sh`. Needs a 24 GB+ card (L4 is just under).",
        ]
    if engine == "ninfer":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `ninfer-qwen3.8-27b-nvfp4` | Qwen3.8-27B NVFP4 + MTP x3 |",
            "| `ninfer-qwen3.8-27b` | Qwen3.8-27B group-wise int + MTP x3 |",
            "| `ninfer-qwen3.6-35b-a3b` | Qwen3.6-35B-A3B MoE + MTP x3 |",
            "| `ninfer-qwen3.8-flash-next` | Flash-Next 101 GiB, CPU expert offload |",
            "",
            "Official: [Neroued/ninfer](https://github.com/Neroued/ninfer) `ninfer-serve`. "
            "Builds only for Blackwell sm_120a, so **G4 runtime only**; first run builds it with CUDA 13 "
            "and caches the build on Drive.",
        ]
    if engine == "basalt":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `basalt-flash-next-nvfp4-dq6k` | Flash-Next NVFP4-DQ6_K, one 132 GB `.basalt` file, MTP drafting |",
            "",
            "Official: [jesdga95/basalt](https://github.com/jesdga95/basalt) `bin/basalt serve MODEL.basalt`. "
            "Builds only for Blackwell sm_120, so **G4 runtime only**; first run builds it with CUDA 13 "
            "and caches the build on Drive.",
        ]
    if engine == "strata_nvfp4":
        return [
            "| Catalog id | Notes |",
            "|---|---|",
            "| `strata-nvfp4-flash-next-gptq` | Flash-Next NVFP4 GPTQ (setup's default family), 129 GB |",
            "",
            "Official: [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) `./setup.sh` then "
            "`serve/server.py --engine strata`. **G4 runtime only** here; the first start runs the fork's setup "
            "(it compiles the engine with CUDA 13) and caches the result on Drive.",
        ]
    return [
        "| Catalog id | Notes |",
        "|---|---|",
        "| `qwen-image-2.1-nf4` | ≤24 GB |",
        "| `qwen-image-2.1` | BF16 on A100/G4 |",
        "| `flux-schnell`, `sdxl-*`, `sd15` | Other Diffusers entries |",
        "| MiniMax H3 (video) | Use the **ComfyUI** notebook (`minimax-h3-*-comfy`) |",
    ]


def _model_choices(engine):
    """Dropdown options for the model picker: "recommended" first, then this engine's catalog ids."""
    return ["recommended"] + [e["id"] for e in catalog.for_engines(catalog.load(), [engine])]


def build_notebook(engine: str, runtime: str | None = None, model: str | None = None) -> dict:
    """With runtime + model: that model's notebook (notebooks/<runtime>/<engine>/<model>.ipynb). Without: an
    engine notebook with a MODEL dropdown (not written to the repo)."""
    meta = ENGINE_META[engine]
    notebook = model_notebook(runtime, engine, model) if model else INDEX_NOTEBOOK
    rt = RUNTIMES[runtime] if runtime else None
    title = meta["title"]
    is_image = engine in ("diffusers", "comfyui")
    kind_label = "image" if is_image else "chat"

    cells = [
        _md([
            _badge(notebook),
            "",
            f"# {model} · {title} · {rt['label']}" if model else f"# Colab {title} hub",
            "",
            f"One authenticated OpenAI-compatible API for **{title}** models on a "
            f"**Google AI Pro / Colab Pro** GPU.",
            f"**This notebook installs only `{engine}`** and advertises only that engine's catalog ids.",
            "",
            (f"- Model: **{catalog.by_id(catalog.load())[model]['name']}** (`{model}`)" if model
             else f"- Supported models: {meta['summary']}"),
            f"- Sibling notebooks: {SIBLING_NOTEBOOKS}",
            "- Drive root: `MyDrive/colab_model_server/` (shared weights/builds; settings file is "
            f"`settings/settings_{engine}.json`)",
            "- Engine matrix: `notebook_code/ENGINE_MATRIX.md`",
            "",
            (f"Runtime → Change runtime type → **{rt['label']}**, then Runtime → Run all."
             if model else "Runtime → Change runtime type → **L4 (RTX 4070 Ti class), A100 or G4 (RTX PRO 6000)**, then run the cells in order."),
        ]),
        _md([
            "## 1. Mount Drive and create the workspace",
            "",
            "The hub's code is packed into this notebook (the collapsed cell below): nothing is downloaded "
            "from GitHub. Settings, models and engine builds are restored from Drive.",
        ]),
        payload_cell(),
        _code([
            "import os, re, shutil, subprocess, sys",
            "from google.colab import drive",
            "drive.mount(\"/content/drive\")",
            "",
            f'ENGINE   = "{engine}"  # locked for this notebook',
            'DRIVE_DIR = "/content/drive/MyDrive/colab_model_server"',
            'HUB_REF   = HUB_CODE_VERSION  # the packed code version (recorded in settings)',
            "assert os.path.isdir(f\"{HUB_DIR}/notebook_code\"), \"Run the hub code cell above first.\"",
            "os.makedirs(DRIVE_DIR, exist_ok=True)",
            "",
            "from notebook_code.workspace import open_workspace",
            "from notebook_code import install",
            "WS = open_workspace(DRIVE_DIR, engine=ENGINE)",
            "print(\"Drive workspace:\", WS.describe())",
            "import notebook_code",
            "print(\"hub\", notebook_code.__version__, \"(code\", HUB_CODE_VERSION + \")\", \"engine\", ENGINE)",
        ]),
        _md(["## 2. Detect GPU, CPU, and RAM — then list this engine's models"]),
        _code([
            "from notebook_code import catalog, gpu, recommend",
            "",
            "G = gpu.detect()[0]",
            "HOST = gpu.print_runtime(G)  # GPU + CPU + RAM + disk, every run",
            "CPU, RAM = HOST[\"cpu\"], HOST[\"ram\"]",
            "VRAM = G[\"vram_gb\"]",
            "ARCH = G[\"arch\"]  # compute capability without the dot: 89 L4, 80 A100, 120 G4",
            "RAM_GB = RAM[\"total_gb\"]",
            "from notebook_code.store import model_disk_gb",
            "DISK_GB = model_disk_gb(WS.local_models)   # free space + cached local models (they are evictable)",
            "ALL = catalog.load()",
            "ENTRIES = catalog.for_engines(ALL, [ENGINE])",
            "PLAN = recommend.for_engine(ENGINE, VRAM, ALL, ram_gb=RAM_GB, disk_gb=DISK_GB, arch=ARCH)",
            "print()",
            "print(f\"Models are read from Google Drive (cached on local disk as they load); SSD-streamed files \"",
            "      f\"are copied to {WS.local_models} ({DISK_GB} GB usable). Fit checks use VRAM, RAM and that disk.\")",
            "print(recommend.explain(VRAM, ALL, engine=ENGINE, ram_gb=RAM_GB, disk_gb=DISK_GB, arch=ARCH))",
            "print()",
            "print(f\"{'model id':36s} {'kind':5s} {'size':>7s}  {'ctx/edge':>14s}\")",
            "for e in ENTRIES:",
            "    reason = catalog.why_not(e, VRAM, RAM_GB, DISK_GB, ARCH)",
            "    fit = reason is None",
            "    kind = catalog.kind_for(e)",
            "    label = \"edge\" if kind in (\"image\", \"video\") else \"ctx\"",
            "    ctx = f\"{catalog.context_for(e, VRAM):,} {label}\" if fit else \"-\"",
            "    mark = \" *\" if e[\"id\"] in PLAN[\"prefetch\"] else \"\"",
            "    print(f\"{e['id']:36s} {kind:5s} {e['size_gb']:6.1f}G  {ctx:>14s}{mark}\" + "
            "(\"\" if fit else f\"   (does not fit: {reason})\"))",
            ("print(\"\\n* = recommended for this GPU + engine.\")" if model else
             "print(\"\\n* = recommended for this GPU + engine. Pick ONE model in the next cell.\")"),
        ]),
        _md([
            f"## 3. Config (saved on Drive under `settings/settings_{engine}.json`)",
            "",
            (f"This notebook serves **`{model}`**: the only model this session downloads and serves. "
             "Other settings come from Drive; edit and re-run to persist." if model else
             "**Pick one model** in the `MODEL` dropdown: it is the only model this session downloads and "
             "serves. `recommended` uses the first recommended model that fits this GPU. Other settings "
             "come from Drive; edit and re-run to persist."),
        ]),
        _code([
            "from notebook_code import catalog, gpu, recommend",
            "",
            "G = gpu.detect()[0]",
            "HOST = gpu.print_runtime(G)  # re-detect GPU + CPU + RAM every config run",
            "CPU, RAM = HOST[\"cpu\"], HOST[\"ram\"]",
            "VRAM = G[\"vram_gb\"]",
            "ARCH = G[\"arch\"]  # compute capability without the dot: 89 L4, 80 A100, 120 G4",
            "RAM_GB = RAM[\"total_gb\"]",
            "from notebook_code.store import model_disk_gb",
            "DISK_GB = model_disk_gb(WS.local_models)   # free space + cached local models (they are evictable)",
            "ALL = catalog.load()",
            "ENTRIES = catalog.for_engines(ALL, [ENGINE])",
            "PLAN = recommend.for_engine(ENGINE, VRAM, ALL, ram_gb=RAM_GB, disk_gb=DISK_GB, arch=ARCH)",
            "saved = WS.load_settings()",
            "",
            "# The ONE model this session downloads to Drive and serves. Nothing else is fetched.",
            (f"MODEL = \"{model}\"  # this notebook's model (notebooks/<runtime>/<engine>/<model>.ipynb)" if model
             else f"MODEL = \"recommended\"  # @param {json.dumps(_model_choices(engine))}"),
            *([f"if PLAN[\"band\"] != \"{rt['band']}\":",
               f"    print(\"Note: this notebook is meant for the {rt['label']} runtime; this one is\", G[\"name\"])"]
              if model else []),
            "SERVE_ONLY_SELECTED = True  # @param {type:\"boolean\"}",
            "fitting = [e[\"id\"] for e in ENTRIES if catalog.fits(e, VRAM, RAM_GB, DISK_GB, ARCH)]",
            "if MODEL == \"recommended\":",
            "    MODEL = (PLAN[\"prefetch\"] or fitting or [None])[0]",
            "    print(\"Recommended model for this GPU:\", MODEL)",
            "assert MODEL, f\"No {ENGINE} model fits this GPU ({VRAM} GB); try a larger runtime.\"",
            "assert MODEL in {e[\"id\"] for e in ENTRIES}, f\"{MODEL} is not a {ENGINE} model\"",
            "assert MODEL in fitting, (f\"{MODEL} does not fit this runtime: \"",
            "                         f\"{catalog.why_not(catalog.by_id(ENTRIES)[MODEL], VRAM, RAM_GB, DISK_GB, ARCH)}\")",
            "PREFETCH = [MODEL]",
            "ENGINES = [ENGINE]  # never install other engines from this notebook",
            "CTX_OVERRIDE = saved.get(\"ctx_override\") or {}",
            "LLAMA_CPP_REF = saved.get(\"llama_cpp_ref\") or \"4c3c5810f87efe177949f1b85ae509d38a2a311f\"",
            "REBUILD_LLAMA = bool(saved.get(\"rebuild_llama\", False))",
            "REFRESH_WHEELS = bool(saved.get(\"refresh_wheels\", False))",
            "# The tunnel is chosen in cell 6 when it starts (a prompt there); this is only the last choice.",
            "TUNNEL = saved.get(\"tunnel\") or \"tailscale\"",
            "CF_HOSTNAME = saved.get(\"cf_hostname\") or \"\"  # cloudflare-named only, e.g. llm.example.com",
            "REFRESH_CF_TOKEN = bool(saved.get(\"refresh_cloudflare_token\", False))",
            "PORT = int(saved.get(\"port\") or 8000)",
            "TS_HOSTNAME = saved.get(\"ts_hostname\") or f\"colab-{ENGINE}\"",
            "TS_FUNNEL = bool(saved.get(\"ts_funnel\", False))  # False = tailnet-only Serve",
            "REFRESH_TS_KEY = bool(saved.get(\"refresh_tailscale_key\", False))",
            "",
            "WS.save_settings({",
            "    \"engine\": ENGINE, \"model\": MODEL, \"serve_only_selected\": SERVE_ONLY_SELECTED,",
            "    \"engines\": ENGINES,",
            "    \"ctx_override\": CTX_OVERRIDE, \"llama_cpp_ref\": LLAMA_CPP_REF,",
            "    \"rebuild_llama\": REBUILD_LLAMA, \"refresh_wheels\": REFRESH_WHEELS,",
            "    \"tunnel\": TUNNEL, \"port\": PORT, \"hub_ref\": HUB_REF,",
            "    \"ts_hostname\": TS_HOSTNAME, \"ts_funnel\": TS_FUNNEL, \"cf_hostname\": CF_HOSTNAME,",
            "    \"refresh_cloudflare_token\": False,",
            "    \"refresh_tailscale_key\": False,  # set True once to re-prompt and overwrite Drive key",
            "    \"gpu_name\": G[\"name\"], \"vram_gb\": VRAM, \"band\": PLAN[\"band\"],",
            "    \"cpu_model\": CPU[\"model\"], \"cpu_cores\": CPU[\"cores\"], \"cpu_threads\": CPU[\"threads\"],",
            "    \"ram_gb\": RAM[\"total_gb\"], \"ram_available_gb\": RAM[\"available_gb\"],",
            "})",
            "print(\"Saved settings →\", WS.settings_path)",
            "print(\"MODEL    =\", MODEL, \"(the only model downloaded and served)\" if SERVE_ONLY_SELECTED else \"(prefetched; others load on request)\")",
            "print(\"ENGINE   =\", ENGINE)",
            "print(\"Tailscale key file (private Drive, not git):\", WS.hub / \"tailscale_authkey.txt\")",
        ]),
        _md([f"## 4. Install {title} (built or downloaded once, then restored from Drive)"]),
        _code([
            "from notebook_code import install",
            "",
            "BUILDS = str(WS.builds)",
            "LLAMA_BIN = \"llama-server\"",
            "STRATA_BIN = \"strata\"",
            "MAYA_ROOT = \"/content/project-maya\"",
            "TENSORFOLD_BIN = \"tensorfold-native\"",
            "COMFYUI_ROOT = \"/content/ComfyUI\"",
            "SPARSA_ROOT = \"/content/Sparsa\"",
            "HYPERQWEN_ROOT = install.HYPERQWEN_ROOT",
            "NINFER_BIN = install.NINFER_ROOT + \"/apps/ninfer-serve\"",
            "BASALT_ROOT = install.BASALT_ROOT",
            "STRATA_NVFP4_ROOT = install.STRATA_NVFP4_ROOT",
            "install.ensure_hub_deps(BUILDS, refresh=REFRESH_WHEELS)",
            f'if ENGINE == "llama":',
            "    LLAMA_BIN = install.ensure_llama(BUILDS, G[\"arch\"], ref=LLAMA_CPP_REF, rebuild=REBUILD_LLAMA)",
            "    print(\"llama-server:\", LLAMA_BIN)",
            f'elif ENGINE == "vllm":',
            "    print(\"vLLM\", install.ensure_vllm(BUILDS, refresh=REFRESH_WHEELS))",
            f'elif ENGINE == "sglang":',
            "    print(\"SGLang venv python:\", install.ensure_sglang(BUILDS, refresh=REFRESH_WHEELS))",
            f'elif ENGINE == "diffusers":',
            "    print(\"diffusers\", install.ensure_diffusers(BUILDS, refresh=REFRESH_WHEELS))",
            f'elif ENGINE == "strata":',
            "    STRATA_BIN = install.ensure_strata(BUILDS, refresh=REFRESH_WHEELS)",
            "    print(\"strata:\", STRATA_BIN)",
            f'elif ENGINE == "maya":',
            "    MAYA_ROOT = install.ensure_maya(BUILDS, refresh=REFRESH_WHEELS)",
            "    print(\"maya:\", MAYA_ROOT)",
            f'elif ENGINE == "tensorfold":',
            "    TENSORFOLD_BIN = install.ensure_tensorfold(BUILDS, refresh=REFRESH_WHEELS)",
            "    print(\"tensorfold:\", TENSORFOLD_BIN)",
            f'elif ENGINE == "comfyui":',
            "    COMFYUI_ROOT = install.ensure_comfyui(BUILDS, refresh=REFRESH_WHEELS)",
            "    print(\"comfyui:\", COMFYUI_ROOT)",
            f'elif ENGINE == "sparsa":',
            "    SPARSA_ROOT = install.ensure_sparsa(BUILDS, refresh=REFRESH_WHEELS)",
            "    print(\"sparsa:\", SPARSA_ROOT)",
            f'elif ENGINE == "hyperqwen":',
            "    HYPERQWEN_ROOT = install.ensure_hyperqwen(BUILDS, refresh=REFRESH_WHEELS)",
            "    print(\"hyperqwen:\", HYPERQWEN_ROOT)",
            f'elif ENGINE == "ninfer":',
            "    NINFER_BIN = install.ensure_ninfer(BUILDS, G[\"arch\"], refresh=REFRESH_WHEELS)",
            "    print(\"ninfer-serve:\", NINFER_BIN)",
            f'elif ENGINE == "basalt":',
            "    BASALT_ROOT = install.ensure_basalt(BUILDS, G[\"arch\"], refresh=REFRESH_WHEELS)",
            "    print(\"basalt:\", BASALT_ROOT)",
            f'elif ENGINE == "strata_nvfp4":',
            "    STRATA_NVFP4_ROOT = install.ensure_strata_nvfp4(BUILDS, G[\"arch\"], refresh=REFRESH_WHEELS)",
            "    print(\"strata-nvfp4:\", STRATA_NVFP4_ROOT)",
            "print(\"Install log →\", WS.logs / \"install.log\")",
        ]),
        _md([
            "## 5. Download the models to Google Drive (skipped for anything already saved)",
            "",
            f"To spend less GPU time, run `{PREPARE_NOTEBOOK}` on a CPU runtime first: it "
            "downloads the model and builds this engine on Drive, and this cell and cell 4 then just restore them.",
        ]),
        _code([
            "import threading, time",
            "from notebook_code.store import ModelStore",
            "",
            "try:",
            "    from google.colab import userdata",
            "    os.environ[\"HF_TOKEN\"] = userdata.get(\"HF_TOKEN\")",
            "except Exception:",
            "    pass",
            "",
            "store = ModelStore(DRIVE_DIR, str(WS.local_models))",
            "for mid in PREFETCH:",
            "    e = catalog.by_id(ENTRIES)[mid]",
            "    if store.state(e)[\"on_drive\"]:",
            "        print(f\"{mid}: already on Drive\")",
            "        continue",
            "    stop = threading.Event()",
            "    def report(mid=mid, e=e):",
            "        while not stop.wait(15):",
            "            p = store.progress.get(mid, {})",
            "            print(f\"  {mid}: {p.get('phase', '')} {p.get('done_bytes', 0) / 1e9:.1f} / {e['size_gb']:.1f} GB\")",
            "    threading.Thread(target=report, daemon=True).start()",
            "    t0 = time.time()",
            "    try:",
            "        store.prefetch(e)",
            "    finally:",
            "        stop.set()",
            "    print(f\"{mid}: saved to Drive in {(time.time() - t0) / 60:.1f} min\")",
        ]),
        _md([
            "## 6. Start the hub and open the tunnel",
            "",
            "Re-running this cell restarts the hub. Secrets stay on **your private Google Drive** "
            "(`MyDrive/colab_model_server/access/`), never in this public GitHub repo or in settings JSON:",
            "",
            "- `access/api_key` — hub Bearer token (auto-created; `access/api_key.txt` is the same key, viewable on a phone)",
            "- `access/tailscale_authkey.txt` — Tailscale auth key (asked once, then reused; open or edit it in the Drive app)",
            "- `access/cloudflare_tunnel_token.txt` — only for the named Cloudflare tunnel (asked once, then reused)",
            "",
            "**Tailscale** (default): create a **reusable + ephemeral** key at "
            "[Tailscale Keys](https://login.tailscale.com/admin/settings/keys) and join the machine your "
            "harness runs on to the same tailnet.",
            "",
            "**Keys, the easy way: Colab Secrets.** Click the key icon in Colab's left sidebar and add "
            "`TAILSCALE_AUTHKEY` (and `HF_TOKEN`, and `CLOUDFLARE_TUNNEL_TOKEN` for a named Cloudflare tunnel). "
            "They belong to your Google account, so every notebook uses them; allow this notebook access once "
            "when Colab asks. A secret wins over the copy on Drive.",
            "",
            "**When this cell runs it asks two things in its output, so you never edit code:** which tunnel "
            "(type 1, 2 or 3; Enter keeps the last one), then any key or token Drive does not have yet, in a "
            "hidden input box. Type e.g. `1 new` at the tunnel prompt to replace a saved key.",
            "",
            "**Cloudflare quick tunnel** (choice 2): nothing to set up, like Unsloth Studio's notebook. "
            "The URL is public and changes every session, so update your harness each time; the hub's API key "
            "is what keeps others off your GPU. The named tunnel (choice 3) uses one you created in the Cloudflare "
            "dashboard (public hostname → `http://127.0.0.1:8000`) for a stable URL.",
            "",
            "Your harness runs on **your own machine**, not here: point it at the Base URL this cell prints.",
        ]),
        _code([
            "import json, requests, time",
            "from notebook_code import keyfile, tunnel, provider",
            "",
            "subprocess.run(\"pkill -TERM -f '[n]otebook_code serve'; "
            "pkill -TERM -f '[t]ailscaled.*colab-llm-tailscale'\", shell=True)",
            "time.sleep(3)",
            "",
            "API_KEY = keyfile.load_or_create(str(WS.hub / \"api_key\"))",
            "HUB_LOG = str(WS.logs / f\"hub_{ENGINE}.log\")",
            "args = [sys.executable, \"-m\", \"notebook_code\", \"serve\", \"--drive-dir\", DRIVE_DIR,",
            "        \"--local-dir\", str(WS.local_models), \"--log-dir\", str(WS.engine_logs),",
            "        \"--port\", str(PORT), \"--llama-bin\", LLAMA_BIN,",
            "        \"--strata-bin\", STRATA_BIN, \"--maya-root\", MAYA_ROOT,",
            "        \"--tensorfold-bin\", TENSORFOLD_BIN, \"--comfyui-root\", COMFYUI_ROOT,",
            "        \"--sparsa-root\", SPARSA_ROOT,",
            "        \"--hyperqwen-root\", HYPERQWEN_ROOT, \"--ninfer-bin\", NINFER_BIN,",
            "        \"--basalt-root\", BASALT_ROOT, \"--strata-nvfp4-root\", STRATA_NVFP4_ROOT,",
            "        \"--engines\", ENGINE]",
            "if SERVE_ONLY_SELECTED:",
            "    args += [\"--models\", MODEL]   # the harness cannot trigger downloads of other models",
            "for k, v in CTX_OVERRIDE.items():",
            "    args += [\"--ctx\", f\"{k}={v}\"]",
            "hub_proc = subprocess.Popen(args, cwd=HUB_DIR, env=dict(os.environ, PYTHONPATH=HUB_DIR, COLAB_LLM_DRIVE=DRIVE_DIR),",
            "                            stdout=open(HUB_LOG, \"w\"), stderr=subprocess.STDOUT)",
            "",
            "AUTH = {\"Authorization\": f\"Bearer {API_KEY}\"}",
            "LAST = \"no answer yet\"",
            "for i in range(90):",
            "    try:",
            "        r = requests.get(f\"http://127.0.0.1:{PORT}/health\", headers=AUTH, timeout=5,",
            "                         proxies={\"http\": None, \"https\": None})",
            "        if r.status_code == 200:",
            "            print(\"Hub is up.\")",
            "            break",
            "        LAST = f\"HTTP {r.status_code}: {r.text[:200]}\"",
            "    except requests.RequestException as e:",
            "        LAST = f\"{type(e).__name__}: {str(e)[:200]}\"",
            "    if hub_proc.poll() is not None:",
            "        raise RuntimeError(\"The hub exited:\\n\" + open(HUB_LOG).read()[-3000:])",
            "    if i in (10, 30, 60):",
            "        print(f\"Still waiting for the hub ({i} s); last answer: {LAST}\")",
            "    time.sleep(1)",
            "else:",
            "    raise RuntimeError(f\"The hub did not answer /health (last: {LAST}):\\n\" + open(HUB_LOG).read()[-3000:])",
            "",
            "# Pick the tunnel in the prompt below (Enter = the last one). Keys and tokens are asked for in a hidden",
            "# input box when Drive has none yet, then kept on private Drive (access/), never in this notebook or git.",
            "TUNNEL, NEW_KEY = tunnel.choose_tunnel(saved.get(\"tunnel\"))",
            "if TUNNEL == \"cloudflare-named\":",
            "    CF_HOSTNAME = tunnel.ask_hostname(CF_HOSTNAME or None)",
            "WS.save_settings({**WS.load_settings(), \"tunnel\": TUNNEL, \"cf_hostname\": CF_HOSTNAME})",
            "if TUNNEL == \"tailscale\":",
            "    TS_AUTHKEY = tunnel.load_or_ask_tailscale_authkey(str(WS.hub / \"tailscale_authkey\"),",
            "                                                      refresh=REFRESH_TS_KEY or NEW_KEY)",
            "    tunnel_proc, BASE = tunnel.start_tailscale(",
            "        PORT, TS_AUTHKEY, hostname=TS_HOSTNAME, drive_dir=DRIVE_DIR, funnel=TS_FUNNEL)",
            "    del TS_AUTHKEY  # drop from notebook namespace; Drive file remains private",
            "elif TUNNEL == \"cloudflare\":",
            "    tunnel_proc, BASE = tunnel.start_cloudflare_quick(PORT, drive_dir=DRIVE_DIR, api_key=API_KEY)",
            "elif TUNNEL == \"cloudflare-named\":",
            "    CF_TOKEN = tunnel.load_or_ask_secret(",
            "        str(WS.hub / \"cloudflare_tunnel_token\"), \"Cloudflare tunnel token (Zero Trust → Networks → \"",
            "        \"Tunnels → your tunnel → install command): \", refresh=REFRESH_CF_TOKEN or NEW_KEY,",
            "        env_name=\"CLOUDFLARE_TUNNEL_TOKEN\")",
            "    tunnel_proc, BASE = tunnel.start_cloudflare_named(PORT, CF_TOKEN, CF_HOSTNAME, drive_dir=DRIVE_DIR,",
            "                                                      api_key=API_KEY)",
            "    del CF_TOKEN",
            "else:",
            "    raise ValueError(f\"Unknown TUNNEL {TUNNEL!r}\")",
            "",
            "# Serving from here on: release this GPU runtime after this many minutes with no API request,",
            "# no cell running and no GPU / CPU / network work, so it stops using compute units (0 = never).",
            "# Setup (installs, downloads) is never timed out: the clock only starts now.",
            "IDLE_TIMEOUT_MIN = 10  # @param {type:\"integer\"}",
            "from notebook_code import idle_guard",
            "if idle_guard.start(IDLE_TIMEOUT_MIN):",
            "    print(f\"Idle timeout: this runtime is released after {IDLE_TIMEOUT_MIN} idle minutes.\")",
            "",
            "MODELS = requests.get(f\"http://127.0.0.1:{PORT}/v1/models\", headers=AUTH).json()[\"data\"]",
            "WS.save_endpoint(BASE + \"/v1\", API_KEY, G[\"name\"], [m[\"id\"] for m in MODELS])",
            "",
            "print(\"Engine   :\", ENGINE)",
            "print(\"Tunnel   :\", TUNNEL)",
            "print(\"Base URL :\", BASE + \"/v1\")",
            "print(\"API key  :\", API_KEY)",
            "print(f\"Models   : {', '.join(m['id'] for m in MODELS)}\")",
            "print(\"Endpoint →\", WS.hub / \"endpoint.json\")",
            "if TUNNEL == \"tailscale\" and not TS_FUNNEL:",
            "    print(\"Reach this URL from any device on your Tailscale tailnet (e.g. the machine your harness runs on).\")",
            "elif TUNNEL == \"cloudflare\":",
            "    print(\"Public URL, new every session: update your harness's base URL each time. The API key protects it.\")",
            "print()",
            "# Your Mac only needs these settings: nothing to install or run there besides your harness.",
            "DSH_YAML = provider.dsh_provider_yaml(BASE, MODELS, provider_id=f\"colab-{ENGINE}\")",
            "(WS.hub / f\"dsh_{ENGINE}.yaml\").write_text(DSH_YAML)",
            "print(\"=== dsh on your Mac ===\")",
            "print(\"1. Add this provider to your dsh settings.yaml (if the file already has 'llm-pi-ai:' / 'providers:',\")",
            "print(f\"   paste from the 'colab-{ENGINE}:' line down, keeping the indentation). Also saved to Drive: \"",
            "      f\"colab_model_server/access/dsh_{ENGINE}.yaml\")",
            "print()",
            "print(DSH_YAML)",
            "print(\"2. Give dsh the key (it is never written into settings.yaml), e.g. in ~/.zshrc:\")",
            "print(f\"   export COLAB_LLM_API_KEY='{API_KEY}'\")",
            "print(\"3. Restart dsh and pick the provider. The URL stays the same every session, so this is one-time.\")",
            "print()",
            "print(\"=== Any other OpenAI-compatible client ===\")",
            "print(provider.openai_env(BASE, API_KEY))",
        ]),
    ]
    if not is_image:
        cells += [
            _md([
                "## 6b. Optional: the dsh harness in this notebook (web UI)",
                "",
                "Instead of (or as well as) connecting a harness on your own machine, this cell can run **dsh** "
                "(the same `@deepseek-ai/dsh` the reference setup uses) right here, already pointed at this "
                "runtime's models, and open its web UI, like the Unsloth Studio notebook opens its own.",
                "",
                "- It asks `y`/`n` when it runs (Enter keeps your last answer); nothing to edit.",
                "- The link opens through **Colab's own proxy**, which only you (signed in to Colab) can open; with "
                "the Tailscale tunnel it is also on your tailnet. dsh also requires the token in the link.",
                "- Conversations are kept on Drive (`dsh/`), and the agent works in Drive's `workspace/` folder, "
                "so both survive the runtime. The agent can run commands **on this Colab runtime**.",
                "- Talking to dsh counts as activity, so the idle timeout does not release the runtime mid-task.",
            ]),
            _code([
                "from notebook_code import dsh_web",
                "from IPython.display import HTML, display",
                "",
                "OPEN_DSH = dsh_web.ask_open(saved.get(\"dsh_web\"))",
                "WS.save_settings({**WS.load_settings(), \"dsh_web\": OPEN_DSH})",
                "if OPEN_DSH:",
                "    DSH_ROOT = dsh_web.ensure(str(WS.builds))",
                "    DSH_PATCH = str(WS.config / f\"dsh_{ENGINE}.patch.yml\")   # no secrets: the key comes from the env",
                "    DSH_MODEL = dsh_web.write_patch(DSH_PATCH, MODELS, PORT, default_model=MODEL)",
                "    PROXY_URL = dsh_web.colab_proxy_url()",
                "    HOSTS = [dsh_web.host_of(PROXY_URL)] if PROXY_URL else []",
                "    TS_DSH_URL = None",
                "    if TUNNEL == \"tailscale\":",
                "        TS_DSH_URL = tunnel.tailscale_serve_port(dsh_web.PORT, dsh_web.TS_HTTPS_PORT, BASE, drive_dir=DRIVE_DIR)",
                "        if TS_DSH_URL:",
                "            HOSTS.append(dsh_web.host_of(TS_DSH_URL))",
                "    DSH_PROC, DSH_TOKEN = dsh_web.start(DSH_ROOT, str(WS.drive / \"dsh\"), str(WS.drive / \"workspace\"),",
                "                                        DSH_PATCH, API_KEY, public_url=PROXY_URL, trusted_hosts=HOSTS)",
                "    print(f\"dsh is running with {DSH_MODEL} (provider 'Colab (this runtime)'). Its log: /content/dsh_web.log\")",
                "    for label, url in ((\"Open dsh (Colab proxy, only you)\", PROXY_URL), (\"Open dsh on your tailnet\", TS_DSH_URL)):",
                "        if url:",
                "            link = f\"{url}?token={DSH_TOKEN}\"",
                "            display(HTML(f'<p><a href=\"{link}\" target=\"_blank\">{label}</a></p>'))",
                "            print(label + \":\", link)",
                "else:",
                "    dsh_web.stop()",
                "    print(\"dsh not started: connect your own harness with the settings printed above.\")",
            ]),
        ]

    if is_image:
        cells += [
            _md([
                "## 7. Try image generation",
                "",
                "The first request loads the pipeline (Drive→local copy, then GPU). "
                "Non-streaming may return `503` + `Retry-After` while loading — re-run the cell.",
            ]),
            _code([
                "from IPython.display import display, Image as IPyImage",
                "import base64",
                "",
                "by = catalog.by_id(ENTRIES)",
                "image_models = [m for m in MODELS if m[\"id\"] in by and catalog.kind_for(by[m[\"id\"]]) == \"image\"]",
                "if not image_models:",
                "    print(\"No image model picked; skipping.\")",
                "else:",
                "",
                "    IMG = next((m for m in PREFETCH if m in {x[\"id\"] for x in image_models}), image_models[0][\"id\"])",
                "    prompt = \"a watercolor fox under pine trees, soft light\"",
                "    t0 = time.time()",
                "    r = requests.post(f\"http://127.0.0.1:{PORT}/v1/images/generations\", headers=AUTH, timeout=900,",
                "                      json={\"model\": IMG, \"prompt\": prompt, \"size\": \"768x768\", \"n\": 1})",
                "    if r.status_code == 503 and r.json().get(\"error\", {}).get(\"code\") == \"model_loading\":",
                "        print(\"Image model still loading; re-run this cell in a minute.\")",
                "    else:",
                "        j = r.json()",
                "        print(\"=== image:\", IMG, j.get(\"size\"), \"===\")",
                "        if r.ok:",
                "            display(IPyImage(data=base64.b64decode(j[\"data\"][0][\"b64_json\"])))",
                "            path = WS.save_image_b64(j[\"data\"][0][\"b64_json\"], IMG, prompt=prompt)",
                "            print(f\"[{time.time() - t0:.1f} s] Saved →\", path)",
                "        else:",
                "            print(j)",
            ]),
            _md([
                "## 7b. Try video generation (MiniMax H3)",
                "",
                "Runs only when a video model is picked. A 5 s clip takes minutes (longer on an L4); "
                "the MP4 is saved to Drive (`outputs/images`).",
            ]),
            _code([
                "from IPython.display import Video",
                "video_models = [m for m in MODELS if m[\"id\"] in by and catalog.kind_for(by[m[\"id\"]]) == \"video\"]",
                "if not video_models:",
                "    print(\"No video model picked; skipping.\")",
                "else:",
                "    VID = next((m for m in PREFETCH if m in {x[\"id\"] for x in video_models}), video_models[0][\"id\"])",
                "    vprompt = \"a red fox trotting through fresh snow at dawn, soft light, natural sound\"",
                "    t0 = time.time()",
                "    r = requests.post(f\"http://127.0.0.1:{PORT}/v1/videos/generations\", headers=AUTH, timeout=7200,",
                "                      json={\"model\": VID, \"prompt\": vprompt, \"size\": \"832x480\", \"seconds\": 5})",
                "    if r.status_code == 503 and r.json().get(\"error\", {}).get(\"code\") == \"model_loading\":",
                "        print(\"Video model still loading; re-run this cell in a minute.\")",
                "    elif r.ok:",
                "        j = r.json()",
                "        path = WS.save_video_b64(j[\"data\"][0][\"b64_mp4\"], VID, prompt=vprompt)",
                "        print(f\"[{time.time() - t0:.0f} s] {j.get('size')}, {j.get('seconds')} s → {path}\")",
                "        display(Video(str(path), embed=True, width=640))",
                "    else:",
                "        print(r.status_code, r.text[:1000])",
            ]),
        ]
    else:
        cells += [
            _md([
                f"## 7. Try {kind_label}",
                "",
                "The first request to a model loads it (download or Drive→local copy, then GPU load). "
                "Streaming chat waits with keep-alives; non-streaming may return `503` + `Retry-After` "
                "while loading — re-run the cell.",
            ]),
            _code([
                "MODEL = next((m for m in PREFETCH if m in {x[\"id\"] for x in MODELS}), MODELS[0][\"id\"])",
                "messages = [{\"role\": \"user\", \"content\": \"In two sentences, what is a model hub?\"}]",
                "t0 = time.time()",
                "r = requests.post(f\"http://127.0.0.1:{PORT}/v1/chat/completions\", headers=AUTH, timeout=900,",
                "                  json={\"model\": MODEL, \"max_tokens\": 200, \"messages\": messages})",
                "if r.status_code == 503 and r.json().get(\"error\", {}).get(\"code\") == \"model_loading\":",
                "    print(\"Chat model still loading; re-run this cell in a minute.\")",
                "else:",
                "    j = r.json()",
                "    text = j[\"choices\"][0][\"message\"][\"content\"] if r.ok else j",
                "    print(\"=== chat:\", MODEL, \"===\")",
                "    print(text)",
                "    print(f\"[{time.time() - t0:.1f} s]\")",
                "    if r.ok:",
                "        path = WS.save_chat(MODEL, messages, text, j.get(\"usage\"))",
                "        print(\"Saved →\", path)",
            ]),
        ]

    cells += [
        _md(["## 8. Status and logs (re-run any time)"]),
        _code([
            "st = requests.get(f\"http://127.0.0.1:{PORT}/v1/hub/status\", headers=AUTH).json()",
            "WS.save_json(WS.status, f\"hub-status-{ENGINE}\", st)",
            "print(json.dumps({k: st[k] for k in (\"gpu\", \"vram_gb\", \"active\", \"loading\", \"phase\", \"inflight\", \"errors\", \"local_free_gb\")}, indent=1))",
            "for mid, p in st[\"downloads\"].items():",
            "    print(f\"  {mid}: {p.get('phase')} {p.get('done_bytes', 0) / 1e9:.1f} / {p.get('total_bytes', 0) / 1e9:.1f} GB\")",
            "print(\"\\n--- hub log (Drive) ---\"); print(open(HUB_LOG).read()[-1500:])",
            "!nvidia-smi --query-gpu=memory.used,memory.total --format=csv",
            "print(\"Outputs →\", WS.outputs)",
        ]),
        _md(["## 9. Stop"]),
        _code([
            "hub_proc.terminate()        # stops the hub and the engine it started",
            "try:",
            "    from notebook_code import dsh_web",
            "    dsh_web.stop()            # the in-notebook harness, if it was started",
            "except Exception:",
            "    pass",
            "try:",
            "    tunnel_proc.terminate()",
            "except Exception:",
            "    pass",
            "if TUNNEL == \"tailscale\":",
            "    tunnel.stop_tailscale(drive_dir=DRIVE_DIR)",
            "else:",
            "    tunnel.stop_cloudflare()",
        ]),
        _md([
            "## Notes",
            "",
            f"### This notebook = `{engine}` only",
            "",
            *_model_table(engine),
            "",
            "### Connecting your harness",
            "",
            "**Tailscale** (choice 1, default, private):",
            "",
            "1. Create a **reusable + ephemeral** auth key: "
            "https://login.tailscale.com/admin/settings/keys",
            "2. First run: cell 6 asks for it in a hidden input box and saves it to **private Drive** "
            "`MyDrive/colab_model_server/access/tailscale_authkey.txt` (not GitHub).",
            "3. Later Colab sessions reuse that Drive file automatically.",
            "4. Easier: add it once as the Colab secret `TAILSCALE_AUTHKEY` (key icon in the left sidebar); "
            "every notebook then uses it without asking.",
            "5. Join the machine your harness runs on to the same tailnet and point the harness at the "
            "printed `https://….ts.net/v1` URL with the API key. The harness (dsh or any OpenAI client) "
            "runs there, never in this notebook.",
            "",
            "**Cloudflare quick tunnel** (choice 2): no account or key. Cell 6 prints a public "
            "`https://….trycloudflare.com/v1` URL once the hub answers through it; it changes every session, and "
            "anyone with it reaches the hub, which still requires the API key. Quick tunnels are a free "
            "best-effort Cloudflare service with no uptime guarantee.",
            "",
            "**Cloudflare named tunnel** (choice 3): create a tunnel in the Cloudflare "
            "dashboard (Zero Trust → Networks → Tunnels), add a public hostname pointing to "
            "`http://127.0.0.1:8000`; cell 6 then asks for the hostname and, once, the tunnel token "
            "(saved to `access/cloudflare_tunnel_token` on private Drive).",
            "",
            "### Drive layout",
            "",
            "| Path under `MyDrive/colab_model_server` | Contents |",
            "|---|---|",
            "| `models/<id>/` | Weights shared across engine notebooks |",
            "| `engines/` | llama.cpp tarballs, wheels, Tailscale, cloudflared |",
            f"| `settings/settings_{engine}.json` | Selected model / ctx / Tailscale for **this** engine (no secrets) |",
            "| `access/api_key`, `access/api_key.txt` | Hub Bearer token (private Drive only; the .txt copy opens on phones) |",
            "| `access/tailscale_authkey.txt` | Tailscale auth key (private Drive only; never in git; opens on phones) |",
            "| `access/cloudflare_tunnel_token.txt` | Cloudflare named-tunnel token (private Drive only; never in git) |",
            "| `access/endpoint.json` | Last base URL + key hint |",
            "| `huggingface/`, `logs/`, `outputs/` | Cache, logs, generation results |",
            "",
            "- See `notebook_code/ENGINE_MATRIX.md` for workstation (Maya / Strata / HyperQwen) notes.",
            "- Set `REFRESH_WHEELS` / `REBUILD_LLAMA` / `refresh_tailscale_key` in settings to force updates.",
            "- **Security:** Tailscale Serve is reachable only from your tailnet (`ts_funnel` makes it public); "
            "Cloudflare URLs are public. Either way the hub API key protects GPU use. Never commit anything "
            "under `access/`.",
        ]),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def build_prepare_notebook() -> dict:
    """CPU-runtime notebook: download a model and build its engine on Drive for a GPU runtime."""
    from . import runtimes
    boot = json.loads(json.dumps(build_notebook("llama")["cells"][1:4]))      # same bootstrap as the GPU notebooks
    src = next(c["source"] for c in boot if any(l.startswith("ENGINE   =") for l in c["source"]))
    i = next(k for k, line in enumerate(src) if line.startswith("ENGINE   ="))
    src[i] = 'ENGINE   = "prepare"  # settings go to settings/settings_prepare.json\n'
    all_ids = ["recommended"] + [e["id"] for e in catalog.load()]
    cells = [
        _md([
            _badge(PREPARE_NOTEBOOK),
            "",
            "# Prepare a model and its engine on a cheap CPU runtime",
            "",
            "Runtime → Change runtime type → **CPU** (High-RAM is not needed). This notebook does the slow, "
            "GPU-free work so the expensive GPU runtime only loads and serves:",
            "",
            "- **Model:** downloaded from Hugging Face straight to `MyDrive/colab_model_server/models/<id>/`, one file "
            "at a time (local disk only ever holds two files). Re-running resumes; files already on Drive are "
            "skipped. Google Drive accepts about 750 GB of uploads per day, so the largest models may need two days.",
            "- **Engine:** built or cached on Drive for the GPU you pick: wheels (vLLM, SGLang, Diffusers), "
            "HyperQwen's patched venv, llama.cpp / NInfer compiled for that GPU's architecture (nvcc does not "
            "need the GPU), release downloads (Strata, Maya, TensorFold, ComfyUI, Sparsa).",
            "- Both run at the same time: the download uses the network, the build uses the CPU.",
            "",
            "Then open `notebooks/<runtime>/<engine>/<model>.ipynb` for the same model on the GPU runtime: it "
            "restores both from Drive.",
            "",
            "Can also be run without a browser: "
            "`python -m notebook_code prepare --target g4 --engine ninfer --model recommended`.",
        ]),
        *boot,
        _md([
            "## 2. Check this runtime",
            "",
            "Compiling llama.cpp or NInfer is limited by CPU cores, and the GPU runtime restores Python wheels "
            "and HyperQwen's venv only if its Python version matches this one.",
        ]),
        _code([
            "import platform, shutil",
            "print(\"CPU threads :\", os.cpu_count())",
            "print(\"Python      :\", platform.python_version(), \"(the GPU runtime must match for wheels / venvs)\")",
            "print(\"Free disk   :\", round(shutil.disk_usage(\"/content\").free / 1e9), \"GB in /content\")",
            "nvcc = subprocess.run(\"PATH=$PATH:/usr/local/cuda/bin nvcc --version | tail -2\", shell=True,",
            "                      capture_output=True, text=True).stdout.strip()",
            "print(\"nvcc        :\", nvcc or \"not found (fine: llama.cpp is then built by the GPU notebook)\")",
            "if shutil.which(\"nvidia-smi\"):",
            "    print(\"Note: this runtime has a GPU. Preparing works here too, but a CPU runtime costs far less.\")",
        ]),
        _md(["## 3. Pick the GPU runtime, engine and model"]),
        _code([
            "from notebook_code import catalog, prepare, runtimes",
            "",
            f"TARGET = \"g4\"  # @param {json.dumps(list(runtimes.TARGETS))}",
            f"ENGINE_TO_PREP = \"vllm\"  # @param {json.dumps(list(ENGINES))}",
            f"MODEL = \"recommended\"  # @param {json.dumps(all_ids)}",
            "BUILD_ENGINE = True  # @param {type:\"boolean\"}",
            "DOWNLOAD_MODEL = True  # @param {type:\"boolean\"}",
            "REFRESH_ENGINE = False  # @param {type:\"boolean\"}",
            "# llama.cpp / NInfer compile for 30+ minutes on a 2-thread CPU runtime and a few on the GPU runtime,",
            "# so they are left to the GPU notebook unless this is ticked.",
            "FORCE_COMPILE = False  # @param {type:\"boolean\"}",
            "",
            "VRAM, RAM_GB, DISK_GB, ARCH = runtimes.specs(TARGET)",
            "print(f\"Target: {runtimes.target(TARGET)['name']} ({VRAM} GB VRAM, {RAM_GB} GB RAM, sm_{ARCH})\\n\")",
            "for e in catalog.for_engines(catalog.load(), [ENGINE_TO_PREP]):",
            "    reason = catalog.why_not(e, VRAM, RAM_GB, DISK_GB, ARCH)",
            "    print(f\"  {e['id']:40s} {e['size_gb']:7.1f} GB  \" + (\"ok\" if reason is None else reason))",
            "ENTRY = prepare.pick(ENGINE_TO_PREP, TARGET, MODEL)",
            "print(f\"\\nWill prepare {ENTRY['id']} ({ENTRY['size_gb']:.1f} GB) for {ENGINE_TO_PREP} on {TARGET}\")",
        ]),
        _md([
            "## 4. Download the model and build the engine",
            "",
            "Safe to stop and re-run: finished files and builds are kept on Drive. Progress prints every 30 s.",
        ]),
        _code([
            "import threading",
            "from notebook_code.store import ModelStore",
            "",
            "try:",
            "    from google.colab import userdata",
            "    os.environ[\"HF_TOKEN\"] = userdata.get(\"HF_TOKEN\")",
            "except Exception:",
            "    pass",
            "",
            "store = ModelStore(DRIVE_DIR, prepare.DEFAULT_STAGING, token=os.environ.get(\"HF_TOKEN\"))",
            "done = threading.Event()",
            "def report():",
            "    while not done.wait(30):",
            "        p = store.progress.get(ENTRY[\"id\"])",
            "        if p:",
            "            print(f\"  {ENTRY['id']}: {p['done_bytes'] / 1e9:.1f} / {p['total_bytes'] / 1e9:.1f} GB on Drive\")",
            "threading.Thread(target=report, daemon=True).start()",
            "try:",
            "    RESULT = prepare.prepare(TARGET, ENGINE_TO_PREP, ENTRY[\"id\"], drive_dir=DRIVE_DIR, store=store,",
            "                             do_engine=BUILD_ENGINE, do_download=DOWNLOAD_MODEL, refresh=REFRESH_ENGINE,",
            "                             force_compile=FORCE_COMPILE)",
            "finally:",
            "    done.set()",
            "print(RESULT)",
            "print(\"Install log →\", WS.logs / \"install.log\")",
        ]),
        _md([
            "## 5. Next",
            "",
            "Disconnect this runtime (Runtime → Disconnect and delete runtime) so it stops using compute units. "
            "Then switch to the GPU runtime and open `notebooks/<runtime>/<engine>/<model>.ipynb` for the same model.",
        ]),
    ]
    return {
        "cells": cells,
        "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def tree_rows(prefix=""):
    """Markdown tables of every notebook, by runtime then engine. prefix: path from the reader to notebooks/."""
    out = []
    for rt, r in RUNTIMES.items():
        out += ["", f"### {r['label']} — `{rt}/`", "", "| Engine | Model notebook |", "|---|---|"]
        for eng in ENGINES:
            models = [m for (x, e, m) in notebook_plan() if x == rt and e == eng]
            if models:
                links = ", ".join(f"[`{m}`]({prefix}{rt}/{eng}/{m}.ipynb)" for m in models)
                out.append(f"| **{ENGINE_META[eng]['title']}** (`{eng}/`) | {links} |")
    return out


def build_index() -> dict:
    cells = [
        _md([
            _badge(INDEX_NOTEBOOK),
            "",
            "# Colab notebooks: runtime → engine → model",
            "",
            "`notebooks/<runtime>/<engine>/<model>.ipynb`. Pick the runtime you have (L4 = RTX 4070 Ti class, A100 80 GB, G4 = RTX PRO 6000), then "
            "the engine, then the model. Each notebook installs only its engine, serves only its model through "
            "one authenticated OpenAI-compatible API, and keeps weights and builds on "
            "`MyDrive/colab_model_server/`.",
            *tree_rows(),
            "",
            f"**Save GPU time:** run [`{PREPARE_NOTEBOOK}`](cpu/prepare_cpu.ipynb) on a cheap CPU runtime "
            "first. It downloads the model to Drive and builds the engine for your GPU, so the GPU notebook "
            "only restores them. See also `notebook_code/ENGINE_MATRIX.md`.",
        ]),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


README_HEAD = """# Colab notebooks: runtime → engine → model

`<runtime>/<engine>/<model>.ipynb`. Open the notebook for your runtime and model in Colab, set the runtime
type it names, and Run all. It installs only that engine, serves only that model (one authenticated
OpenAI-compatible API), and keeps the weights and engine builds on Google Drive.
"""

README_TAIL = """
## Other

- [`cpu/prepare_cpu.ipynb`](cpu/prepare_cpu.ipynb): on a cheap CPU runtime, download a model to Drive and build
  its engine for the GPU you will use, so the GPU notebook only restores them.
- [`other/unsloth/`](other/unsloth/): official Unsloth Studio.

Regenerate every notebook from the repo root (the hub code is packed into each one):

```bash
python3 -m notebook_code.make_engine_notebooks
```
"""


def write_all(root: Path | None = None):
    root = Path(root or ROOT)
    written = []
    for rt, eng, model in notebook_plan():
        path = root / model_notebook(rt, eng, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_notebook(eng, rt, model), indent=1) + "\n")
        written.append(path)
    # A model dropped from a runtime's list loses its notebook.
    for rt in RUNTIMES:
        for old in (root / "notebooks" / rt).glob("*/*.ipynb"):
            if old not in written:
                old.unlink()
    prep = root / PREPARE_NOTEBOOK
    prep.parent.mkdir(parents=True, exist_ok=True)
    prep.write_text(json.dumps(build_prepare_notebook(), indent=1) + "\n")
    written.append(prep)
    index = root / INDEX_NOTEBOOK
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(build_index(), indent=1) + "\n")
    written.append(index)
    (root / "notebooks" / "README.md").write_text(README_HEAD + "\n".join(tree_rows()) + "\n" + README_TAIL)
    return written


def main():
    for path in write_all():
        print("wrote", path)


if __name__ == "__main__":
    main()
