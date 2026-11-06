"""Generate one Colab hub notebook per engine.

Run from the repo root:
  python3 -m colab_llm_hub.make_engine_notebooks
"""
from __future__ import annotations

import json
from pathlib import Path

from . import catalog
from .recommend import ENGINE_META, ENGINES, INDEX_NOTEBOOK

REPO = "https://github.com/Biegle-W/google_colab_llm"
BRANCH = "main"
ROOT = Path(__file__).resolve().parents[1]


def _src(lines):
    out = []
    for line in lines:
        out.append(line if line.endswith("\n") else line + "\n")
    return out


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


SIBLING_NOTEBOOKS = (
    "`notebooks/llama/`, `notebooks/vllm/`, `notebooks/sglang/`, `notebooks/diffusers/`, "
    "`notebooks/strata/`, `notebooks/maya/`, `notebooks/tensorfold/`, `notebooks/comfyui/`, "
    "`notebooks/sparsa/` — see `notebooks/llm_hub_colab.ipynb`"
)


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
            "| `qwen-image-2.1-turbo-comfy` | Workstation Comfy image entry |",
            "| `flux3-klein-comfy` | FLUX.3 Klein |",
            "| `wan22-5b-comfy`, `minimax-h3-comfy` | Video prefetch (OpenAI image route returns 501) |",
            "",
            "UI graphs under `colab_llm_hub/workflows/*.ui.json` (custom NvidiaResident nodes are optional).",
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
    return [
        "| Catalog id | Notes |",
        "|---|---|",
        "| `qwen-image-2.1-nf4` | ≤24 GB |",
        "| `qwen-image-2.1` | BF16 on A100/G4 |",
        "| `flux-schnell`, `sdxl-*`, `sd15` | Other Diffusers entries |",
        "| `minimax-h3` | Prefetch only (video); prefer **ComfyUI** notebook |",
    ]


def _model_choices(engine):
    """Dropdown options for the model picker: "recommended" first, then this engine's catalog ids."""
    return ["recommended"] + [e["id"] for e in catalog.for_engines(catalog.load(), [engine])]


def build_notebook(engine: str) -> dict:
    meta = ENGINE_META[engine]
    notebook = meta["notebook"]
    title = meta["title"]
    is_image = engine in ("diffusers", "comfyui")
    kind_label = "image" if is_image else "chat"

    cells = [
        _md([
            _badge(notebook),
            "",
            f"# Colab {title} hub",
            "",
            f"One authenticated OpenAI-compatible API for **{title}** models on a "
            f"**Google AI Pro / Colab Pro** GPU.",
            f"**This notebook installs only `{engine}`** and advertises only that engine's catalog ids.",
            "",
            f"- Supported models: {meta['summary']}",
            f"- Sibling notebooks: {SIBLING_NOTEBOOKS}",
            "- Drive root: `MyDrive/llm_models/` (shared weights/builds; settings file is "
            f"`config/settings_{engine}.json`)",
            "- Engine matrix: `colab_llm_hub/ENGINE_MATRIX.md`",
            "",
            "Runtime → Change runtime type → **L4, A100 or G4**, then run the cells in order.",
        ]),
        _md([
            "## 1. Mount Drive, create the workspace, fetch (or restore) the hub code",
            "",
            "Each run fetches the latest hub code from GitHub and refreshes its copy on Drive; "
            "only if GitHub cannot be reached does it fall back to that Drive copy. Settings and "
            "tool binaries are restored from Drive.",
        ]),
        _code([
            "import os, re, shutil, subprocess, sys",
            "from google.colab import drive",
            "drive.mount(\"/content/drive\")",
            "",
            f'ENGINE   = "{engine}"  # locked for this notebook',
            f'HUB_REPO  = "{REPO}"',
            f'HUB_REF   = "main"  # always clone hub package from main',
            'DRIVE_DIR = "/content/drive/MyDrive/llm_models"',
            'HUB_DIR   = "/content/hub_repo"',
            "",
            "def sh(cmd):",
            "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)",
            "    if r.returncode:",
            "        raise RuntimeError(f\"{cmd}\\n{r.stdout}{r.stderr}\")",
            "    return r.stdout.strip()",
            "",
            "OFFLINE_HUB = False   # True = skip GitHub and use the hub code saved on Drive",
            "",
            "os.makedirs(DRIVE_DIR, exist_ok=True)",
            "sys.path.insert(0, HUB_DIR)",
            "ref_key = (HUB_REF or \"default\").replace(\"/\", \"_\")",
            "cached = f\"{DRIVE_DIR}/builds/hub_pkg/{ref_key}/colab_llm_hub\"",
            "have_cache = os.path.isdir(cached) and os.path.isfile(f\"{DRIVE_DIR}/builds/hub_pkg/{ref_key}/COMPLETE\")",
            "fresh = False",
            "if not OFFLINE_HUB:",
            "    try:                          # always prefer the latest code: fixes land without reopening",
            "        sh(f\"rm -rf {HUB_DIR}\")",
            "        if HUB_REF and re.fullmatch(r\"[0-9a-f]{40}\", HUB_REF):",
            "            sh(f\"mkdir -p {HUB_DIR} && cd {HUB_DIR} && git init -q && "
            "git fetch -q --depth 1 {HUB_REPO} {HUB_REF} && git checkout -q FETCH_HEAD\")",
            "        else:",
            "            sh(f\"git clone -q --depth 1 {'--branch ' + HUB_REF if HUB_REF else ''} "
            "{HUB_REPO} {HUB_DIR}\")",
            "        fresh = True",
            "        print(\"Fetched hub code from GitHub:\", sh(f\"cd {HUB_DIR} && git rev-parse --short HEAD\"))",
            "    except RuntimeError as e:",
            "        print(\"Could not fetch from GitHub; using the copy on Drive.\", str(e)[-300:])",
            "if not fresh:",
            "    if not have_cache:",
            "        raise RuntimeError(\"No hub code: GitHub was unreachable and there is no copy on Drive yet.\")",
            "    shutil.rmtree(HUB_DIR, ignore_errors=True)",
            "    os.makedirs(HUB_DIR, exist_ok=True)",
            "    shutil.copytree(cached, f\"{HUB_DIR}/colab_llm_hub\")",
            "    print(\"Restored hub package from Drive (it may be older than GitHub)\")",
            "",
            "from colab_llm_hub.workspace import open_workspace",
            "from colab_llm_hub import install",
            "WS = open_workspace(DRIVE_DIR, engine=ENGINE)",
            "print(\"Drive workspace:\", WS.describe())",
            "if fresh:                         # keep the Drive copy current for offline fallback",
            "    install.cache_hub_package(HUB_DIR, str(WS.builds), ref=HUB_REF or \"default\")",
            "import colab_llm_hub",
            "print(\"hub\", colab_llm_hub.__version__, \"engine\", ENGINE)",
        ]),
        _md(["## 2. Detect GPU, CPU, and RAM — then list this engine's models"]),
        _code([
            "from colab_llm_hub import catalog, gpu, recommend",
            "",
            "G = gpu.detect()[0]",
            "HOST = gpu.print_runtime(G)  # GPU + CPU + RAM + disk, every run",
            "CPU, RAM = HOST[\"cpu\"], HOST[\"ram\"]",
            "VRAM = G[\"vram_gb\"]",
            "RAM_GB = RAM[\"total_gb\"]",
            "from colab_llm_hub.store import model_disk_gb",
            "DISK_GB = model_disk_gb(WS.local_models)   # free space + cached local models (they are evictable)",
            "ALL = catalog.load()",
            "ENTRIES = catalog.for_engines(ALL, [ENGINE])",
            "PLAN = recommend.for_engine(ENGINE, VRAM, ALL, ram_gb=RAM_GB, disk_gb=DISK_GB)",
            "print()",
            "print(f\"Model files go to {WS.local_models} ({DISK_GB} GB usable); fit checks use VRAM, RAM and that disk.\")",
            "print(recommend.explain(VRAM, ALL, engine=ENGINE, ram_gb=RAM_GB, disk_gb=DISK_GB))",
            "print()",
            "print(f\"{'model id':36s} {'kind':5s} {'size':>7s}  {'ctx/edge':>14s}\")",
            "for e in ENTRIES:",
            "    reason = catalog.why_not(e, VRAM, RAM_GB, DISK_GB)",
            "    fit = reason is None",
            "    kind = catalog.kind_for(e)",
            "    label = \"edge\" if kind in (\"image\", \"video\") else \"ctx\"",
            "    ctx = f\"{catalog.context_for(e, VRAM):,} {label}\" if fit else \"-\"",
            "    mark = \" *\" if e[\"id\"] in PLAN[\"prefetch\"] else \"\"",
            "    print(f\"{e['id']:36s} {kind:5s} {e['size_gb']:6.1f}G  {ctx:>14s}{mark}\" + "
            "(\"\" if fit else f\"   (does not fit: {reason})\"))",
            "print(\"\\n* = recommended for this GPU + engine. Pick ONE model in the next cell.\")",
        ]),
        _md([
            f"## 3. Config (saved on Drive under `config/settings_{engine}.json`)",
            "",
            "**Pick one model** in the `MODEL` dropdown: it is the only model this session downloads and "
            "serves. `recommended` uses the first recommended model that fits this GPU. Other settings "
            "come from Drive; edit and re-run to persist.",
        ]),
        _code([
            "from colab_llm_hub import catalog, gpu, recommend",
            "",
            "G = gpu.detect()[0]",
            "HOST = gpu.print_runtime(G)  # re-detect GPU + CPU + RAM every config run",
            "CPU, RAM = HOST[\"cpu\"], HOST[\"ram\"]",
            "VRAM = G[\"vram_gb\"]",
            "RAM_GB = RAM[\"total_gb\"]",
            "from colab_llm_hub.store import model_disk_gb",
            "DISK_GB = model_disk_gb(WS.local_models)   # free space + cached local models (they are evictable)",
            "ALL = catalog.load()",
            "ENTRIES = catalog.for_engines(ALL, [ENGINE])",
            "PLAN = recommend.for_engine(ENGINE, VRAM, ALL, ram_gb=RAM_GB, disk_gb=DISK_GB)",
            "saved = WS.load_settings()",
            "",
            "# The ONE model this session downloads to Drive and serves. Nothing else is fetched.",
            f"MODEL = \"recommended\"  # @param {json.dumps(_model_choices(engine))}",
            "SERVE_ONLY_SELECTED = True  # @param {type:\"boolean\"}",
            "fitting = [e[\"id\"] for e in ENTRIES if catalog.fits(e, VRAM, RAM_GB, DISK_GB)]",
            "if MODEL == \"recommended\":",
            "    MODEL = (PLAN[\"prefetch\"] or fitting or [None])[0]",
            "    print(\"Recommended model for this GPU:\", MODEL)",
            "assert MODEL, f\"No {ENGINE} model fits this GPU ({VRAM} GB); try a larger runtime.\"",
            "assert MODEL in {e[\"id\"] for e in ENTRIES}, f\"{MODEL} is not a {ENGINE} model\"",
            "assert MODEL in fitting, (f\"{MODEL} does not fit this runtime: \"",
            "                         f\"{catalog.why_not(catalog.by_id(ENTRIES)[MODEL], VRAM, RAM_GB, DISK_GB)}\")",
            "PREFETCH = [MODEL]",
            "ENGINES = [ENGINE]  # never install other engines from this notebook",
            "CTX_OVERRIDE = saved.get(\"ctx_override\") or {}",
            "LLAMA_CPP_REF = saved.get(\"llama_cpp_ref\") or \"4c3c5810f87efe177949f1b85ae509d38a2a311f\"",
            "REBUILD_LLAMA = bool(saved.get(\"rebuild_llama\", False))",
            "REFRESH_WHEELS = bool(saved.get(\"refresh_wheels\", False))",
            "# Connection: Tailscale only. The auth key is stored on private Drive (hub/tailscale_authkey), never in git.",
            "TUNNEL = \"tailscale\"",
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
            "    \"ts_hostname\": TS_HOSTNAME, \"ts_funnel\": TS_FUNNEL,",
            "    \"refresh_tailscale_key\": False,  # set True once to re-prompt and overwrite Drive key",
            "    \"gpu_name\": G[\"name\"], \"vram_gb\": VRAM, \"band\": PLAN[\"band\"],",
            "    \"cpu_model\": CPU[\"model\"], \"cpu_cores\": CPU[\"cores\"], \"cpu_threads\": CPU[\"threads\"],",
            "    \"ram_gb\": RAM[\"total_gb\"], \"ram_available_gb\": RAM[\"available_gb\"],",
            "})",
            "print(\"Saved settings →\", WS.settings_path)",
            "print(\"MODEL    =\", MODEL, \"(the only model downloaded and served)\" if SERVE_ONLY_SELECTED else \"(prefetched; others load on request)\")",
            "print(\"ENGINE   =\", ENGINE)",
            "print(\"TUNNEL   =\", TUNNEL)",
            "print(\"Tailscale key file (private Drive, not git):\", WS.hub / \"tailscale_authkey\")",
        ]),
        _md([f"## 4. Install {title} (built or downloaded once, then restored from Drive)"]),
        _code([
            "from colab_llm_hub import install",
            "",
            "BUILDS = str(WS.builds)",
            "LLAMA_BIN = \"llama-server\"",
            "STRATA_BIN = \"strata\"",
            "MAYA_ROOT = str(WS.builds / \"project-maya\")",
            "TENSORFOLD_BIN = \"tensorfold-native\"",
            "COMFYUI_ROOT = str(WS.builds / \"ComfyUI\")",
            "SPARSA_ROOT = str(WS.builds / \"Sparsa\")",
            "install.ensure_hub_deps(BUILDS, refresh=REFRESH_WHEELS)",
            f'if ENGINE == "llama":',
            "    LLAMA_BIN = install.ensure_llama(BUILDS, G[\"arch\"], ref=LLAMA_CPP_REF, rebuild=REBUILD_LLAMA)",
            "    print(\"llama-server:\", LLAMA_BIN)",
            f'elif ENGINE == "vllm":',
            "    print(\"vLLM\", install.ensure_vllm(BUILDS, refresh=REFRESH_WHEELS))",
            f'elif ENGINE == "sglang":',
            "    print(\"SGLang\", install.ensure_sglang(BUILDS, refresh=REFRESH_WHEELS))",
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
            "print(\"Install log →\", WS.logs / \"install.log\")",
        ]),
        _md([
            "## 5. Download the models to Google Drive (skipped for anything already saved)",
        ]),
        _code([
            "import threading, time",
            "from colab_llm_hub.store import ModelStore",
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
            "## 6. Start the hub and open Tailscale",
            "",
            "Re-running this cell restarts the hub. Secrets stay on **your private Google Drive** "
            "(`MyDrive/llm_models/hub/`), never in this public GitHub repo or in settings JSON:",
            "",
            "- `hub/api_key` — hub Bearer token (auto-created)",
            "- `hub/tailscale_authkey` — Tailscale auth key (prompted once, then reused)",
            "",
            "Create a **reusable + ephemeral** key at "
            "[Tailscale Keys](https://login.tailscale.com/admin/settings/keys). "
            "Set `REFRESH_TS_KEY = True` (or `refresh_tailscale_key` in settings) to overwrite the Drive copy.",
            "",
            "Your harness runs on **your own machine**, not here: join that machine to the same tailnet and "
            "point the harness at the Base URL this cell prints.",
        ]),
        _code([
            "import json, requests, time",
            "from colab_llm_hub import keyfile, tunnel, provider",
            "",
            "subprocess.run(\"pkill -TERM -f '[c]olab_llm_hub serve'; "
            "pkill -TERM -f '[t]ailscaled.*colab-llm-tailscale'\", shell=True)",
            "time.sleep(3)",
            "",
            "API_KEY = keyfile.load_or_create(str(WS.hub / \"api_key\"))",
            "HUB_LOG = str(WS.logs / f\"hub_{ENGINE}.log\")",
            "args = [sys.executable, \"-m\", \"colab_llm_hub\", \"serve\", \"--drive-dir\", DRIVE_DIR,",
            "        \"--local-dir\", str(WS.local_models), \"--log-dir\", str(WS.engine_logs),",
            "        \"--port\", str(PORT), \"--llama-bin\", LLAMA_BIN,",
            "        \"--strata-bin\", STRATA_BIN, \"--maya-root\", MAYA_ROOT,",
            "        \"--tensorfold-bin\", TENSORFOLD_BIN, \"--comfyui-root\", COMFYUI_ROOT,",
            "        \"--sparsa-root\", SPARSA_ROOT,",
            "        \"--engines\", ENGINE]",
            "if SERVE_ONLY_SELECTED:",
            "    args += [\"--models\", MODEL]   # the harness cannot trigger downloads of other models",
            "for k, v in CTX_OVERRIDE.items():",
            "    args += [\"--ctx\", f\"{k}={v}\"]",
            "hub_proc = subprocess.Popen(args, cwd=HUB_DIR, env=dict(os.environ, PYTHONPATH=HUB_DIR, COLAB_LLM_DRIVE=DRIVE_DIR),",
            "                            stdout=open(HUB_LOG, \"w\"), stderr=subprocess.STDOUT)",
            "",
            "AUTH = {\"Authorization\": f\"Bearer {API_KEY}\"}",
            "for _ in range(60):",
            "    try:",
            "        if requests.get(f\"http://127.0.0.1:{PORT}/health\", headers=AUTH, timeout=2).status_code == 200:",
            "            break",
            "    except requests.RequestException:",
            "        pass",
            "    if hub_proc.poll() is not None:",
            "        raise RuntimeError(\"The hub exited:\\n\" + open(HUB_LOG).read()[-3000:])",
            "    time.sleep(1)",
            "else:",
            "    raise RuntimeError(\"The hub did not start:\\n\" + open(HUB_LOG).read()[-3000:])",
            "",
            "# Load the auth key from private Drive, or prompt once and save it there (never committed to git).",
            "TS_KEY_PATH = str(WS.hub / \"tailscale_authkey\")",
            "TS_AUTHKEY = tunnel.load_or_ask_tailscale_authkey(TS_KEY_PATH, refresh=REFRESH_TS_KEY)",
            "tunnel_proc, BASE = tunnel.start_tailscale(",
            "    PORT, TS_AUTHKEY, hostname=TS_HOSTNAME, drive_dir=DRIVE_DIR, funnel=TS_FUNNEL)",
            "del TS_AUTHKEY  # drop from notebook namespace; Drive file remains private",
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
            "if not TS_FUNNEL:",
            "    print(\"Reach this URL from any device on your Tailscale tailnet (e.g. the machine your harness runs on).\")",
            "print()",
            "print(\"Any OpenAI-compatible client:\")",
            "print(provider.openai_env(BASE, API_KEY))",
            "print(\"dsh (runs on YOUR machine, not in this notebook): in a clone of this repo there, run\")",
            "print(f\"  python -m colab_llm_hub.configure_dsh --url {BASE} --key {API_KEY} "
            "--settings <path to dsh settings.yaml>\")",
            "print(f\"  export COLAB_LLM_API_KEY={API_KEY}\\n\")",
            "print(provider.dsh_provider_yaml(BASE, MODELS))",
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
                "    raise RuntimeError(\"No image models visible for this GPU; pick another model in the MODEL dropdown (cell 3).\")",
                "",
                "IMG = next((m for m in PREFETCH if m in {x[\"id\"] for x in image_models}), image_models[0][\"id\"])",
                "prompt = \"a watercolor fox under pine trees, soft light\"",
                "t0 = time.time()",
                "r = requests.post(f\"http://127.0.0.1:{PORT}/v1/images/generations\", headers=AUTH, timeout=900,",
                "                  json={\"model\": IMG, \"prompt\": prompt, \"size\": \"768x768\", \"n\": 1})",
                "if r.status_code == 503 and r.json().get(\"error\", {}).get(\"code\") == \"model_loading\":",
                "    print(\"Image model still loading; re-run this cell in a minute.\")",
                "else:",
                "    j = r.json()",
                "    print(\"=== image:\", IMG, j.get(\"size\"), \"===\")",
                "    if r.ok:",
                "        display(IPyImage(data=base64.b64decode(j[\"data\"][0][\"b64_json\"])))",
                "        path = WS.save_image_b64(j[\"data\"][0][\"b64_json\"], IMG, prompt=prompt)",
                "        print(f\"[{time.time() - t0:.1f} s] Saved →\", path)",
                "    else:",
                "        print(j)",
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
            "    tunnel_proc.terminate()",
            "except Exception:",
            "    pass",
            "if TUNNEL == \"tailscale\":",
            "    tunnel.stop_tailscale(drive_dir=DRIVE_DIR)",
        ]),
        _md([
            "## Notes",
            "",
            f"### This notebook = `{engine}` only",
            "",
            *_model_table(engine),
            "",
            "### Connecting your harness (Tailscale)",
            "",
            "1. Create a **reusable + ephemeral** auth key: "
            "https://login.tailscale.com/admin/settings/keys",
            "2. First run: cell 6 prompts for it and saves to **private Drive** "
            "`MyDrive/llm_models/hub/tailscale_authkey` (not GitHub).",
            "3. Later Colab sessions reuse that Drive file automatically.",
            "4. Optional Colab secret `TAILSCALE_AUTHKEY` if you press Enter on an empty prompt.",
            "5. Join the machine your harness runs on to the same tailnet and point the harness at the "
            "printed `https://….ts.net/v1` URL with the API key. The harness (dsh or any OpenAI client) "
            "runs there, never in this notebook.",
            "",
            "### Drive layout",
            "",
            "| Path under `MyDrive/llm_models` | Contents |",
            "|---|---|",
            "| `models/<id>/` | Weights shared across engine notebooks |",
            "| `builds/` | llama.cpp tarballs, wheels, hub_pkg, Tailscale |",
            f"| `config/settings_{engine}.json` | Selected model / ctx / Tailscale for **this** engine (no secrets) |",
            "| `hub/api_key` | Hub Bearer token (private Drive only) |",
            "| `hub/tailscale_authkey` | Tailscale auth key (private Drive only; never in git) |",
            "| `hub/endpoint.json` | Last base URL + key hint |",
            "| `hf_cache/`, `logs/`, `outputs/` | Cache, logs, generation results |",
            "",
            "- See `colab_llm_hub/ENGINE_MATRIX.md` for workstation (Maya / Strata / HyperQwen) notes.",
            "- Set `REFRESH_WHEELS` / `REBUILD_LLAMA` / `refresh_tailscale_key` in settings to force updates.",
            "- **Security:** Tailscale Serve is reachable only from your tailnet (`ts_funnel` makes it public); "
            "the hub API key still protects GPU use. Never commit `hub/api_key` or `hub/tailscale_authkey`.",
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


def build_index() -> dict:
    rows = [
        "| Notebook | Engine | Models |",
        "|---|---|---|",
    ]
    for eng in ENGINES:
        m = ENGINE_META[eng]
        # Index lives at notebooks/llm_hub_colab.ipynb → link relative to that folder.
        rel = m["notebook"][len("notebooks/"):] if m["notebook"].startswith("notebooks/") else m["notebook"]
        rows.append(f"| [`{m['notebook']}`]({rel}) | **{m['title']}** | {m['summary']} |")
    cells = [
        _md([
            _badge(INDEX_NOTEBOOK),
            "",
            "# Colab GPU notebooks (one per engine)",
            "",
            "Notebooks are grouped under `notebooks/<engine>/`. Each `<engine>_gpu.ipynb` runs the hub on an "
            "L4, A100 or G4: it installs **only that engine**, lists **only its supported models**, and saves "
            "settings under `config/settings_<engine>.json`. Weights and builds still share "
            "`MyDrive/llm_models/`. For a T4 use `llama/llama_cpp_t4.ipynb`; for a TPU, `vllm/vllm_tpu.ipynb`.",
            "",
            *rows,
            "",
            "Older single-model notebooks live in the same engine folders "
            "(`notebooks/llama/llama_cpp_*.ipynb`, `notebooks/vllm/vllm_*.ipynb`, …). "
            "See also `colab_llm_hub/ENGINE_MATRIX.md`.",
            "",
            "Runtime → Change runtime type → **L4, A100 or G4**.",
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


def write_all(root: Path | None = None):
    root = Path(root or ROOT)
    written = []
    for eng in ENGINES:
        path = root / ENGINE_META[eng]["notebook"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_notebook(eng), indent=1) + "\n")
        written.append(path)
    index = root / INDEX_NOTEBOOK
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(build_index(), indent=1) + "\n")
    written.append(index)
    return written


def main():
    for path in write_all():
        print("wrote", path)


if __name__ == "__main__":
    main()
