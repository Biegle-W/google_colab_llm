# Colab LLM + diffusion hub

One authenticated, OpenAI-compatible endpoint in front of **one** inference engine on a Colab GPU
(Google AI Pro / Colab Pro L4, A100, or G4), so a harness on your own computer can use Colab's GPUs for
**chat** or **text-to-image**. Model files live on Google Drive and are downloaded only once.

**One notebook per engine** (each installs only that engine and advertises only its catalog models):

| Notebook | Engine |
|---|---|
| [`llama_hub_colab.ipynb`](../llama_hub_colab.ipynb) | llama.cpp |
| [`vllm_hub_colab.ipynb`](../vllm_hub_colab.ipynb) | vLLM |
| [`sglang_hub_colab.ipynb`](../sglang_hub_colab.ipynb) | SGLang |
| [`diffusers_hub_colab.ipynb`](../diffusers_hub_colab.ipynb) | Diffusers |

[`llm_hub_colab.ipynb`](../llm_hub_colab.ipynb) is an index. Target models and per-GPU picks are in
[`ENGINE_MATRIX.md`](ENGINE_MATRIX.md). Defaults come from `recommend.for_engine(engine, vram_gb)`.
Regenerate notebooks with `python3 -m colab_llm_hub.make_engine_notebooks`.

## How it works

```
your client ──HTTPS──> Cloudflare tunnel ──> hub (127.0.0.1:8000, Bearer key)
                                               ├─ llama.cpp   (:30080)  GGUF chat models
                                               ├─ vLLM        (:30010)  safetensors chat models
                                               ├─ SGLang      (:30020)  chat models
                                               └─ Diffusers   (:30090)  text-to-image pipelines

Drive root MyDrive/llm_models/
  models/  builds/  config/  hf_cache/  hub/  logs/  outputs/
Hugging Face ──once──> local disk ──copy──> Drive models/
later sessions: Drive ──copy──> local ──> engine   (wheels/builds/hub_pkg/cloudflared restored too)
```

| Piece | File |
|---|---|
| Hub server: auth, routing, loading, switching, keep-alives | `hub.py` |
| Model files: download once, save to Drive, restore, evict | `store.py` |
| llama.cpp / vLLM / SGLang / Diffusers command lines and process control | `backends.py` |
| Diffusers OpenAI images server (incl. Qwen-Image-2.1 / NF4) | `diffusers_server.py` |
| Model list, what fits which GPU, context / max edge | `catalog.json`, `catalog.py` |
| Per-engine × GPU recommendations | `recommend.py`, `ENGINE_MATRIX.md` |
| Generate the four `*_hub_colab.ipynb` notebooks | `make_engine_notebooks.py` |
| Drive layout: per-engine settings, HF cache, logs, outputs | `workspace.py` |
| GPU detection (including Blackwell/CUDA handling) | `gpu.py` |
| Engine builds and wheels cached on Drive | `install.py` |
| Cloudflare quick or named tunnel | `tunnel.py` |
| dsh provider block, stable API key | `provider.py`, `keyfile.py` |
| Local script: add the hub to dsh's settings | `configure_dsh.py` |

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
10 s until the first token. **Non-streaming** (and image) requests wait about 85 s (under Cloudflare's roughly 100 s idle
limit) and then get `503` with `Retry-After`; the load keeps going, so a retry succeeds.

## Connecting your harness

The notebook prints the base URL, the API key and a dsh provider block (chat models only). On your own computer, in a clone of this
repository, `python -m colab_llm_hub.configure_dsh --url <URL> --key <KEY> --settings <dsh settings.yaml>` asks the
hub which models it serves and merges one provider into the settings (other providers and settings are left alone,
a `.bak` is written, and the key is never saved in the file: dsh reads it from `COLAB_LLM_API_KEY`). A quick-tunnel
URL changes every session; a Cloudflare named tunnel (see the notebook's notes) gives a fixed one.

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
