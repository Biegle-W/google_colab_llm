# Colab LLM + diffusion hub

One authenticated, OpenAI-compatible endpoint in front of **one** inference engine on a Colab GPU
(Google AI Pro / Colab Pro L4, A100, or G4), so a harness on your own computer can use Colab's GPUs for
**chat** or **text-to-image**. Model files live on Google Drive and are downloaded only once.

**One notebook per runtime, engine and model:** `notebooks/<runtime>/<engine>/<model>.ipynb` for the L4,
A100 80 GB and G4 runtimes (each installs only its engine and serves only its model). The list is
[`notebooks/README.md`](../notebooks/README.md); [`notebooks/llm_hub_colab.ipynb`](../notebooks/llm_hub_colab.ipynb)
is the same index as a notebook. Which models get a notebook: `recommend.notebook_plan()` (each engine's
recommended models per runtime). Per-GPU picks and research are in [`ENGINE_MATRIX.md`](ENGINE_MATRIX.md).
Regenerate with `python3 -m notebook_code.make_engine_notebooks`.

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

Drive root MyDrive/colab_model_server/
  models/  engines/  settings/  huggingface/  access/  logs/  outputs/
Hugging Face ──once, file by file──> Drive models/
every session:  VRAM+RAM models: engine reads Drive directly
                SSD-streamed models: streamed files Drive ──copy──> local disk, the rest read from Drive
                (wheels/builds/Tailscale restored too;
                the hub code itself is packed inside each notebook: nothing is fetched from GitHub)
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
| Generate the `<runtime>/<engine>/<model>.ipynb` notebooks | `make_engine_notebooks.py` |
| Drive layout: per-engine settings, HF cache, logs, outputs | `workspace.py` |
| GPU detection (including Blackwell/CUDA handling) | `gpu.py` |
| Engine builds and wheels cached on Drive | `install.py` |
| Tailscale Serve (tailnet) / Funnel, or Cloudflare quick / named tunnel | `tunnel.py` |
| dsh provider block, stable API key | `provider.py`, `keyfile.py` |

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
to Drive as `access/dsh_<engine>.yaml`. Paste it into dsh's `settings.yaml` and set `COLAB_LLM_API_KEY` (the key is
never written into the settings file). The URL is the same every session, so this is one-time per engine.
Remote access defaults to **Tailscale Serve**: the auth key is prompted once and saved only under the user's
private Drive (`access/tailscale_authkey`, gitignored). `TUNNEL = "cloudflare"` opens a Cloudflare quick tunnel
instead (no account, new public URL each session, returned only once the hub answers through it), and
`cloudflare-named` runs your own dashboard tunnel by token (`access/cloudflare_tunnel_token`). Public URLs are
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
- **Tests.** `python -m unittest discover -s notebook_code/tests -t .` covers auth, model filtering,
  on-demand loading, switching (including chat ↔ image), waiting for in-flight streams, keep-alives, slow loads,
  failed starts, crash recovery, prefetch, Drive save/restore, interrupted copies, eviction, resume of partial
  downloads, Diffusers command lines / size clamping, and merging the provider into dsh settings.
- **Real engine, no GPU.** The real hub, real Hugging Face downloader and the real llama.cpp binary served a
  135M model (non-streaming and streaming), saved it to a fake Drive, and a second "session" with the local disk
  wiped restored it from Drive without downloading.
- **Model repos.** Catalog repo and file names are expected to resolve on Hugging Face; sizes are approximate.

## Validated on Colab GPUs

The notebooks have run on real Colab L4, A100 80 GB (High-RAM) and G4 (RTX PRO 6000) runtimes. Every measured
result (prefill, time to first token, decode, cold start; seconds per image or clip) is in
[`results/results.csv`](../results/results.csv), shown in the main [README](../README.md#tested-on-colab), with the
generated images and videos in [`results/samples/`](../results/samples/). Runtime hardware was measured on those
sessions (VRAM, RAM, disk, including the A100's 369 GB NVMe scratch disk) and is recorded in `recommend.RUNTIMES`.

<!-- UNTESTED:START -->

Notebooks not yet run on their GPU runtime (no row in `results/results.csv`):

- **L4 (RTX 4070 Ti class, 24 GB):** none, all tested.
- **A100 80 GB (High-RAM):** `diffusers/flux-schnell`, `maya/glm-maya-s`, `sparsa/minimax-m3.1-flash-iq2-kt`, `strata/deepseek-strata-ist-2.00`.
- **G4 (RTX PRO 6000 Blackwell, 96 GB):** `comfyui/flux3-klein-comfy`, `comfyui/minimax-h3-g4-comfy`, `comfyui/qwen-image-2.1-bf16-comfy`, `comfyui/qwen-image-2.1-turbo-bf16-comfy`, `comfyui/wan22-5b-comfy`, `diffusers/flux-schnell`, `llama/nemotron-3.5-bolt-74b-iq2`, `llama/ornith-1.5-122b-q4`, `llama/sage-a1.5-q4`, `sparsa/minimax-m3.1-flash-iq2-kt`.

<!-- UNTESTED:END -->

Still estimates rather than measurements:

- Context / max-edge sizes in `catalog.json` are conservative estimates for each amount of VRAM, not validated
  allocations.
- Gated models (FLUX, some Black Forest Labs and NVIDIA repos) need `HF_TOKEN` or `access/hf_token.txt` on Drive,
  and the Hugging Face account must have accepted the model's terms.

## Adding a model

Add an entry to `catalog.json` (the tests validate the file).

- **llama.cpp:** Hugging Face repo, `files` pattern, `model_file`, size, `contexts` as `[[min VRAM GB, tokens], ...]`.
- **vLLM:** repo, parsers, `language_model_only`, same `contexts` shape.
- **Diffusers:** repo, `pipeline` (e.g. `AutoPipelineForText2Image`), `dtype`, `default_steps`, `default_guidance`,
  and `contexts` as `[[min VRAM GB, max edge px], ...]`.

Extra engine flags go in `extra_args`.

