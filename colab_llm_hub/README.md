# Colab LLM + diffusion hub

One authenticated, OpenAI-compatible endpoint in front of **one** inference engine on a Colab GPU
(Google AI Pro / Colab Pro L4, A100, or G4), so a harness on your own computer can use Colab's GPUs for
**chat** or **text-to-image**. Model files live on Google Drive and are downloaded only once.

**One notebook per engine** under [`notebooks/<engine>/`](../notebooks/) (each installs only that
engine and advertises only its catalog models):

| Notebook | Engine |
|---|---|
| [`notebooks/strata/`](../notebooks/strata/) | **Strata** (Flash-Next ISTA/Gyro/Swift, Qwen3.6, DeepSeek) |
| [`notebooks/maya/`](../notebooks/maya/) | **Maya** (GLM-5.3-Flash, DeepSeek) |
| [`notebooks/tensorfold/`](../notebooks/tensorfold/) | **TensorFold** (Qwen3.8-27B EXL3 + DFlash2) |
| [`notebooks/sparsa/`](../notebooks/sparsa/) | **Sparsa** (MiniMax-M3.1-Flash) |
| [`notebooks/hyperqwen/`](../notebooks/hyperqwen/) | **HyperQwen** (Qwen3.8-27B W4A16, patched vLLM 0.32) |
| [`notebooks/ninfer/`](../notebooks/ninfer/) | **NInfer** (native C++/CUDA, G4 only) |
| [`notebooks/basalt/`](../notebooks/basalt/) | **Basalt** (Strata fork for Blackwell, G4 only) |
| [`notebooks/strata_nvfp4/`](../notebooks/strata_nvfp4/) | **Strata NVFP4 fork** (G4 only) |
| [`notebooks/comfyui/`](../notebooks/comfyui/) | **ComfyUI** (Qwen-Image Turbo, FLUX.3 Klein, video prefetch) |
| [`notebooks/llama/`](../notebooks/llama/) | llama.cpp (Bolt IQ2, Sage, Ornith, MoE fallbacks) |
| [`notebooks/vllm/`](../notebooks/vllm/) | vLLM |
| [`notebooks/sglang/`](../notebooks/sglang/) | SGLang |
| [`notebooks/diffusers/`](../notebooks/diffusers/) | Diffusers |

[`notebooks/llm_hub_colab.ipynb`](../notebooks/llm_hub_colab.ipynb) is an index. Target models and
per-GPU picks are in [`ENGINE_MATRIX.md`](ENGINE_MATRIX.md). Defaults come from
`recommend.for_engine(engine, vram_gb)`. Regenerate with `python3 -m colab_llm_hub.make_engine_notebooks`.

## How it works

```
your machine (harness) ──HTTPS──> Tailscale Serve (tailnet) or Cloudflare tunnel (public) ──> hub (127.0.0.1:8000, Bearer key)
                                               ├─ llama.cpp   (:30080)  GGUF chat models
                                               ├─ vLLM        (:30010)  safetensors chat models
                                               ├─ SGLang      (:30020)  chat models
                                               ├─ HyperQwen   (:30030)  Qwen3.8-27B (patched vLLM)
                                               ├─ NInfer      (:30040)  native C++/CUDA (G4)
                                               ├─ Strata      (:30050)  Flash-Next / Qwen3.6 / DeepSeek
                                               ├─ Maya        (:30060)  GLM / DeepSeek
                                               ├─ Sparsa      (:30065)  MiniMax-M3.1-Flash
                                               ├─ TensorFold  (:30070)  EXL3 + DFlash2
                                               ├─ Diffusers   (:30090)  text-to-image pipelines
                                               └─ ComfyUI     (:30095)  Comfy workflows + OpenAI images

Drive root MyDrive/llm_models/
  models/  builds/  config/  hf_cache/  hub/  logs/  outputs/
Hugging Face ──once, file by file──> Drive models/
every session:  VRAM+RAM models: engine reads Drive directly
                SSD-streamed models: streamed files Drive ──copy──> local disk, the rest read from Drive
                (wheels/builds/Tailscale restored too;
                hub code is fetched from GitHub each run, hub_pkg/ on Drive is only the offline fallback)
```

| Piece | File |
|---|---|
| Hub server: auth, routing, loading, switching, keep-alives | `hub.py` |
| Model files: download once, save to Drive, restore, evict | `store.py` |
| Prepare on a CPU runtime: Drive-direct download + engine build for a target GPU | `prepare.py`, `runtimes.py` |
| Engine command lines and process control | `backends.py` |
| Diffusers OpenAI images server (incl. Qwen-Image-2.1 / NF4) | `diffusers_server.py` |
| ComfyUI OpenAI images wrapper + stock API workflows | `comfyui_server.py`, `workflows/` |
| Model list, what fits which GPU, context / max edge | `catalog.json`, `catalog.py` |
| Per-engine × GPU recommendations | `recommend.py`, `ENGINE_MATRIX.md` |
| Generate the `<engine>_gpu.ipynb` notebooks | `make_engine_notebooks.py` |
| Drive layout: per-engine settings, HF cache, logs, outputs | `workspace.py` |
| GPU detection (including Blackwell/CUDA handling) | `gpu.py` |
| Engine builds and wheels cached on Drive | `install.py` |
| Tailscale Serve (tailnet) / Funnel, or Cloudflare quick / named tunnel | `tunnel.py` |
| dsh provider block, stable API key | `provider.py`, `keyfile.py` |
| Optional local script: merge the provider into dsh's settings | `configure_dsh.py` |

### Endpoints (all need `Authorization: Bearer <key>`)

| Endpoint | |
|---|---|
| `GET /v1/models` | models that fit this GPU (`kind`: `chat` or `image`), with `context_length` / `max_edge`, `loaded`, `downloaded` |
| `POST /v1/chat/completions`, `/v1/completions` | proxied to llama.cpp / vLLM; the `model` field selects it |
| `POST /v1/images/generations` | proxied to Diffusers; body includes `prompt`, optional `size` / `width` / `height` / `steps` / `seed` |
| `GET /v1/hub/status` | active model, load phase, download progress, free disk |
| `POST /v1/hub/prefetch` `{"model": id}` | download a model to Drive in the background |
| `POST /v1/hub/unload` | stop the engine |

Loading behavior: **streaming** chat requests are answered with headers at once and `: loading ...` comment lines every
10 s until the first token. **Non-streaming** (and image) requests wait about 85 s (under common proxy idle
limits) and then get `503` with `Retry-After`; the load keeps going, so a retry succeeds.

## Connecting your harness

Everything runs in Colab; your computer only runs the harness. Cell 6 prints the base URL, the API key and a
ready-to-paste dsh provider block (`colab-<engine>`, chat models with their Colab context sizes), and saves the block
to Drive as `hub/dsh_<engine>.yaml`. Paste it into dsh's `settings.yaml` and set `COLAB_LLM_API_KEY` (the key is
never written into the settings file). The URL is the same every session, so this is one-time per engine.
Optional: `configure_dsh.py` does the same merge automatically if you have this repo and PyYAML on that computer.
Remote access defaults to **Tailscale Serve**: the auth key is prompted once and saved only under the user's
private Drive (`hub/tailscale_authkey`, gitignored). `TUNNEL = "cloudflare"` opens a Cloudflare quick tunnel
instead (no account, new public URL each session, returned only once the hub answers through it), and
`cloudflare-named` runs your own dashboard tunnel by token (`hub/cloudflare_tunnel_token`). Public URLs are
protected by the hub API key. The harness always runs on your own machine; no harness runs inside a notebook. Each notebook serves the one model picked in its `MODEL` dropdown
(`serve --models <id>`), so requests for other models return 404 instead of starting a download.

Image clients call the same base URL:

```bash
curl "$OPENAI_BASE_URL/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"sdxl-turbo","prompt":"a fox in watercolor","size":"768x768"}'
```

## What it was checked against

- **Flags.** Every llama.cpp flag the hub passes was checked against `llama-server --help` of the commit the
  reference setup runs (`4c3c5810`, built CPU-only here), including `--reasoning`, `--chat-template-kwargs`,
  `--cache-ram`, `--ctx-checkpoints`.
- **Tests.** `python -m unittest discover -s colab_llm_hub/tests -t .` covers auth, model filtering,
  on-demand loading, switching (including chat ↔ image), waiting for in-flight streams, keep-alives, slow loads,
  failed starts, crash recovery, prefetch, Drive save/restore, interrupted copies, eviction, resume of partial
  downloads, Diffusers command lines / size clamping, and merging the provider into dsh settings.
- **Real engine, no GPU.** The real hub, real Hugging Face downloader and the real llama.cpp binary served a
  135M model (non-streaming and streaming), saved it to a fake Drive, and a second "session" with the local disk
  wiped restored it from Drive without downloading.
- **Model repos.** Catalog repo and file names are expected to resolve on Hugging Face; sizes are approximate.

## What has **not** been validated

- Nothing here has run on a Colab GPU. No speed, memory or context numbers have been measured.
- Context / max-edge sizes in `catalog.json` are conservative estimates for each amount of VRAM, not validated allocations.
- vLLM support for these model architectures on whichever vLLM pip installs, and on Blackwell (G4), is unverified.
  The Nemotron entry is marked experimental. FP8 on an A100 runs through a weight-only path (no native FP8).
- Diffusers pipelines (SDXL / FLUX) and their VRAM use on Colab images are unverified; gated models need `HF_TOKEN`.
- Colab's runtimes (A100 memory size, whether a G4 is the 96 GB RTX PRO 6000) are taken from public information and
  measured at run time; the catalog adapts to the reported VRAM.

## Adding a model

Add an entry to `catalog.json` (the tests validate the file).

- **llama.cpp:** Hugging Face repo, `files` pattern, `model_file`, size, `contexts` as `[[min VRAM GB, tokens], ...]`.
- **vLLM:** repo, parsers, `language_model_only`, same `contexts` shape.
- **Diffusers:** repo, `pipeline` (e.g. `AutoPipelineForText2Image`), `dtype`, `default_steps`, `default_guidance`,
  and `contexts` as `[[min VRAM GB, max edge px], ...]`.

Extra engine flags go in `extra_args`.
