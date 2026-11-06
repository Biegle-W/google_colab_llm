"""Generate one Colab hub notebook per engine.

Run from the repo root:
  python3 -m colab_llm_hub.make_engine_notebooks
"""
from __future__ import annotations

import json
from pathlib import Path

from .recommend import ENGINE_META, ENGINES

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
    "`llama_hub_colab.ipynb`, `vllm_hub_colab.ipynb`, `sglang_hub_colab.ipynb`, "
    "`diffusers_hub_colab.ipynb`, `strata_hub_colab.ipynb`, `maya_hub_colab.ipynb`, "
    "`tensorfold_hub_colab.ipynb`, `comfyui_hub_colab.ipynb`, `sparsa_hub_colab.ipynb`"
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
            "First run downloads/clones once and caches under Drive. Later runs restore the hub "
            "package, settings, and tool binaries from Drive.",
        ]),
        _code([
            "import os, re, shutil, subprocess, sys",
            "from google.colab import drive",
            "drive.mount(\"/content/drive\")",
            "",
            f'ENGINE   = "{engine}"  # locked for this notebook',
            f'HUB_REPO  = "{REPO}"',
            f'HUB_REF   = "{BRANCH}"  # pin until merged; then set None for default branch',
            'DRIVE_DIR = "/content/drive/MyDrive/llm_models"',
            'HUB_DIR   = "/content/hub_repo"',
            "",
            "def sh(cmd):",
            "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)",
            "    if r.returncode:",
            "        raise RuntimeError(f\"{cmd}\\n{r.stdout}{r.stderr}\")",
            "    return r.stdout.strip()",
            "",
            "os.makedirs(DRIVE_DIR, exist_ok=True)",
            "sys.path.insert(0, HUB_DIR)",
            "ref_key = (HUB_REF or \"default\").replace(\"/\", \"_\")",
            "cached = f\"{DRIVE_DIR}/builds/hub_pkg/{ref_key}/colab_llm_hub\"",
            "if os.path.isdir(cached) and os.path.isfile(f\"{DRIVE_DIR}/builds/hub_pkg/{ref_key}/COMPLETE\"):",
            "    shutil.rmtree(HUB_DIR, ignore_errors=True)",
            "    os.makedirs(HUB_DIR, exist_ok=True)",
            "    shutil.copytree(cached, f\"{HUB_DIR}/colab_llm_hub\")",
            "    print(\"Restored hub package from Drive\")",
            "else:",
            "    sh(f\"rm -rf {HUB_DIR}\")",
            "    if HUB_REF and re.fullmatch(r\"[0-9a-f]{40}\", HUB_REF):",
            "        sh(f\"mkdir -p {HUB_DIR} && cd {HUB_DIR} && git init -q && "
            "git fetch -q --depth 1 {HUB_REPO} {HUB_REF} && git checkout -q FETCH_HEAD\")",
            "    else:",
            "        sh(f\"git clone -q --depth 1 {'--branch ' + HUB_REF if HUB_REF else ''} "
            "{HUB_REPO} {HUB_DIR}\")",
            "",
            "from colab_llm_hub.workspace import open_workspace",
            "from colab_llm_hub import install",
            "WS = open_workspace(DRIVE_DIR, engine=ENGINE)",
            "print(\"Drive workspace:\", WS.describe())",
            "if not (os.path.isdir(cached) and os.path.isfile(f\"{DRIVE_DIR}/builds/hub_pkg/{ref_key}/COMPLETE\")):",
            "    install.cache_hub_package(HUB_DIR, str(WS.builds), ref=HUB_REF or \"default\")",
            "import colab_llm_hub",
            "print(\"hub\", colab_llm_hub.__version__, \"engine\", ENGINE)",
        ]),
        _md(["## 2. Detect the GPU and list this engine's models"]),
        _code([
            "from colab_llm_hub import catalog, gpu, recommend",
            "",
            "G = gpu.detect()[0]",
            "VRAM = G[\"vram_gb\"]",
            "ALL = catalog.load()",
            "ENTRIES = catalog.for_engines(ALL, [ENGINE])",
            "PLAN = recommend.for_engine(ENGINE, VRAM, ALL)",
            "print(f\"{G['name']}: {VRAM} GB VRAM, compute capability {G['compute_cap']} (sm_{G['arch']})\")",
            "!free -h | head -2",
            "!df -h /content | tail -1",
            "print()",
            "print(recommend.explain(VRAM, ALL, engine=ENGINE))",
            "print()",
            "print(f\"{'model id':36s} {'kind':5s} {'size':>7s}  {'ctx/edge':>14s}\")",
            "for e in ENTRIES:",
            "    fit = catalog.fits(e, VRAM)",
            "    kind = catalog.kind_for(e)",
            "    label = \"edge\" if kind in (\"image\", \"video\") else \"ctx\"",
            "    ctx = f\"{catalog.context_for(e, VRAM):,} {label}\" if fit else f\"needs {catalog.min_vram_gb(e)} GB\"",
            "    mark = \" *\" if e[\"id\"] in PLAN[\"prefetch\"] else \"\"",
            "    print(f\"{e['id']:36s} {kind:5s} {e['size_gb']:6.1f}G  {ctx:>14s}{mark}\" + "
            "(\"\" if fit else \"   (does not fit)\"))",
            "print(\"\\n* = recommended prefetch for this GPU + engine\")",
        ]),
        _md([
            f"## 3. Config (saved on Drive under `config/settings_{engine}.json`)",
            "",
            "Defaults come from the engine recommender, then from any settings already on Drive. "
            "Edit and re-run to persist.",
        ]),
        _code([
            "from colab_llm_hub import catalog, gpu, recommend",
            "",
            "G = gpu.detect()[0]",
            "VRAM = G[\"vram_gb\"]",
            "ALL = catalog.load()",
            "ENTRIES = catalog.for_engines(ALL, [ENGINE])",
            "PLAN = recommend.for_engine(ENGINE, VRAM, ALL)",
            "saved = WS.load_settings()",
            "",
            "PREFETCH = saved.get(\"prefetch\") or PLAN[\"prefetch\"]",
            "ENGINES = [ENGINE]  # never install other engines from this notebook",
            "CTX_OVERRIDE = saved.get(\"ctx_override\") or {}",
            "LLAMA_CPP_REF = saved.get(\"llama_cpp_ref\") or \"4c3c5810f87efe177949f1b85ae509d38a2a311f\"",
            "REBUILD_LLAMA = bool(saved.get(\"rebuild_llama\", False))",
            "REFRESH_WHEELS = bool(saved.get(\"refresh_wheels\", False))",
            "# Default: Tailscale (auth key is prompted every fresh run; never stored on Drive).",
            "# Alternatives: set TUNNEL to \"quick\" or \"named\" and keep_cloudflare_quick=True for Cloudflare.",
            "TUNNEL = saved.get(\"tunnel\") or \"tailscale\"",
            "if TUNNEL == \"quick\" and not saved.get(\"keep_cloudflare_quick\"):",
            "    TUNNEL = \"tailscale\"  # migrate older Drive defaults to Tailscale",
            "PORT = int(saved.get(\"port\") or 8000)",
            "TS_HOSTNAME = saved.get(\"ts_hostname\") or f\"colab-{ENGINE}\"",
            "TS_FUNNEL = bool(saved.get(\"ts_funnel\", False))  # False = tailnet-only Serve"
            "",
            "if not PREFETCH:",
            "    # Fall back to the smallest model that fits so the notebook is still runnable.",
            "    fit_ids = [e[\"id\"] for e in ENTRIES if catalog.fits(e, VRAM)]",
            "    PREFETCH = fit_ids[:1]",
            "    print(\"No band default; using\", PREFETCH)",
            "",
            "WS.save_settings({",
            "    \"engine\": ENGINE, \"prefetch\": PREFETCH, \"engines\": ENGINES,",
            "    \"ctx_override\": CTX_OVERRIDE, \"llama_cpp_ref\": LLAMA_CPP_REF,",
            "    \"rebuild_llama\": REBUILD_LLAMA, \"refresh_wheels\": REFRESH_WHEELS,",
            "    \"tunnel\": TUNNEL, \"port\": PORT, \"hub_ref\": HUB_REF,",
            "    \"ts_hostname\": TS_HOSTNAME, \"ts_funnel\": TS_FUNNEL,",
            "    \"gpu_name\": G[\"name\"], \"vram_gb\": VRAM, \"band\": PLAN[\"band\"],",
            "})",
            "print(\"Saved settings →\", WS.settings_path)",
            "print(\"PREFETCH =\", PREFETCH)",
            "print(\"ENGINE   =\", ENGINE)",
            "print(\"TUNNEL   =\", TUNNEL, \"(Tailscale auth key is prompted in the next tunnel cell)\")",
            "unknown = [m for m in PREFETCH if m not in {e[\"id\"] for e in ENTRIES}]",
            "assert not unknown, f\"unknown or wrong-engine model ids: {unknown}\"",
            "not_fit = [m for m in PREFETCH if not catalog.fits(catalog.by_id(ENTRIES)[m], VRAM)]",
            "assert not not_fit, f\"these do not fit this GPU: {not_fit}\"",
            "assert all(catalog.by_id(ENTRIES)[m][\"backend\"] == ENGINE for m in PREFETCH)",
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
            "## 6. Start the hub and open Tailscale (auth key prompted every run)",
            "",
            "Re-running this cell restarts the hub. The **API key** is created once and kept on Drive.",
            "The **Tailscale auth key is never stored on Drive** — you paste it on every fresh Colab runtime.",
            "",
            "Create a **reusable + ephemeral** key at "
            "[Tailscale Keys](https://login.tailscale.com/admin/settings/keys). "
            "Default tunnel is Tailscale Serve (reachable only from your tailnet). "
            "Set `TUNNEL=\"quick\"` / `\"named\"` in settings for Cloudflare instead.",
        ]),
        _code([
            "import json, requests, time",
            "from colab_llm_hub import keyfile, tunnel, provider",
            "",
            "subprocess.run(\"pkill -TERM -f '[c]olab_llm_hub serve'; pkill -f '[c]loudflared'; "
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
            "if TUNNEL == \"tailscale\":",
            "    # Always ask on a fresh run; Enter alone may use Colab secret TAILSCALE_AUTHKEY.",
            "    TS_AUTHKEY = tunnel.ask_tailscale_authkey()",
            "    tunnel_proc, BASE = tunnel.start_tailscale(",
            "        PORT, TS_AUTHKEY, hostname=TS_HOSTNAME, drive_dir=DRIVE_DIR, funnel=TS_FUNNEL)",
            "    del TS_AUTHKEY  # do not keep the auth key in the notebook namespace",
            "elif TUNNEL == \"named\":",
            "    from google.colab import userdata",
            "    tunnel.ensure_cloudflared(drive_dir=DRIVE_DIR)",
            "    tunnel_proc = tunnel.start_named(userdata.get(\"CLOUDFLARE_TUNNEL_TOKEN\"), drive_dir=DRIVE_DIR)",
            "    BASE = \"https://\" + userdata.get(\"CLOUDFLARE_HOSTNAME\").removeprefix(\"https://\").strip(\"/\")",
            "else:",
            "    tunnel.ensure_cloudflared(drive_dir=DRIVE_DIR)",
            "    tunnel_proc, BASE = tunnel.start_quick(PORT, drive_dir=DRIVE_DIR)",
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
            "    print(\"Reach this URL from any device on your Tailscale tailnet.\")",
            "print()",
            "print(\"Any OpenAI-compatible client:\")",
            "print(provider.openai_env(BASE, API_KEY))",
            "print(\"dsh: on your computer, in a clone of this repo:\")",
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
                "    raise RuntimeError(\"No image models visible for this GPU; lower size or pick another Diffusers id in PREFETCH.\")",
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
            "### Remote access (Tailscale by default)",
            "",
            "1. Create a **reusable + ephemeral** auth key: "
            "https://login.tailscale.com/admin/settings/keys",
            "2. Every fresh Colab runtime, cell 6 prompts you to paste it (not saved on Drive).",
            "3. Optional Colab secret `TAILSCALE_AUTHKEY`: press Enter at the prompt to use it.",
            "4. Connect from your PC (on the same tailnet) to the printed `https://….ts.net/v1` URL.",
            "5. Cloudflare fallbacks: set `TUNNEL=\"quick\"` or `\"named\"` in settings.",
            "",
            "### Drive layout",
            "",
            "| Path under `MyDrive/llm_models` | Contents |",
            "|---|---|",
            "| `models/<id>/` | Weights shared across engine notebooks |",
            "| `builds/` | llama.cpp tarballs, wheels, hub_pkg, cloudflared, Tailscale |",
            f"| `config/settings_{engine}.json` | Prefetch / ctx / tunnel for **this** engine |",
            "| `hub/api_key`, `hub/endpoint.json` | Stable API key + last public URL |",
            "| `hf_cache/`, `logs/`, `outputs/` | Cache, logs, generation results |",
            "",
            "- See `colab_llm_hub/ENGINE_MATRIX.md` for workstation (Maya / Strata / HyperQwen) notes.",
            "- Set `REFRESH_WHEELS` / `REBUILD_LLAMA` in settings to force updates.",
            "- **Security:** Tailscale Serve is tailnet-only. Cloudflare quick tunnels are public; "
            "the hub API key still protects GPU use.",
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
        rows.append(f"| `{m['notebook']}` | **{m['title']}** | {m['summary']} |")
    cells = [
        _md([
            _badge("llm_hub_colab.ipynb"),
            "",
            "# Colab inference hubs (one notebook per engine)",
            "",
            "Pick the engine you want. Each notebook installs **only that engine**, lists **only "
            "its supported models**, and saves settings under `config/settings_<engine>.json`. "
            "Weights and builds still share `MyDrive/llm_models/`.",
            "",
            *rows,
            "",
            "Also see the older single-model notebooks (`llama_cpp_gpu.ipynb`, `vllm_gpu.ipynb`, …) "
            "and `colab_llm_hub/ENGINE_MATRIX.md`.",
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
        path.write_text(json.dumps(build_notebook(eng), indent=1) + "\n")
        written.append(path)
    index = root / "llm_hub_colab.ipynb"
    index.write_text(json.dumps(build_index(), indent=1) + "\n")
    written.append(index)
    return written


def main():
    for path in write_all():
        print("wrote", path)


if __name__ == "__main__":
    main()
