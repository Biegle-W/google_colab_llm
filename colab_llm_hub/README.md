# Colab LLM hub

One authenticated, OpenAI-compatible endpoint in front of llama.cpp and vLLM on a Colab GPU, so a harness
running on your own computer (for example dsh) can use Colab's GPUs. It follows the design of the local model hub
in [`Biegle-W/local-llm-public`](https://github.com/Biegle-W/local-llm-public): a request that names a model
starts or switches the right engine; replies in progress are never cut off; streaming clients get keep-alives
while a model loads.

Run it with [`llm_hub_colab.ipynb`](../llm_hub_colab.ipynb). The notebook only fetches this package and calls it,
so changes pushed here take effect the next time the notebook runs.

## How it works

```
your harness ──HTTPS──> Cloudflare tunnel ──> hub (127.0.0.1:8000, Bearer key)
                                                 ├─ llama.cpp  (:30080)  GGUF models
                                                 └─ vLLM       (:30010)  safetensors models
models:  Hugging Face ──once──> Colab local disk ──copy──> Google Drive (MyDrive/llm_models/models/<id>)
later sessions:  Drive ──copy──> local disk ──> engine
```

| Piece | File |
|---|---|
| Hub server: auth, routing, loading, switching, keep-alives | `hub.py` |
| Model files: download once, save to Drive, restore, evict | `store.py` |
| llama.cpp / vLLM command lines and process control | `backends.py` |
| Model list, what fits which GPU, context sizes | `catalog.json`, `catalog.py` |
| GPU detection (including Blackwell/CUDA handling) | `gpu.py` |
| Engine builds cached on Drive | `install.py` |
| Cloudflare quick or named tunnel | `tunnel.py` |
| dsh provider block, stable API key | `provider.py`, `keyfile.py` |
| Local script: add the hub to dsh's settings | `configure_dsh.py` |

### Endpoints (all need `Authorization: Bearer <key>`)

| Endpoint | |
|---|---|
| `GET /v1/models` | models that fit this GPU, with `context_length`, `loaded`, `downloaded` |
| `POST /v1/chat/completions`, `/v1/completions` | proxied to the engine; the `model` field selects it |
| `GET /v1/hub/status` | active model, load phase, download progress, free disk |
| `POST /v1/hub/prefetch` `{"model": id}` | download a model to Drive in the background |
| `POST /v1/hub/unload` | stop the engine |

Loading behavior: **streaming** requests are answered with headers at once and `: loading ...` comment lines every
10 s until the first token. **Non-streaming** requests wait about 85 s (under Cloudflare's roughly 100 s idle
limit) and then get `503` with `Retry-After`; the load keeps going, so a retry succeeds.

## Connecting your harness

The notebook prints the base URL, the API key and a dsh provider block. On your own computer, in a clone of this
repository, `python -m colab_llm_hub.configure_dsh --url <URL> --key <KEY> --settings <dsh settings.yaml>` asks the
hub which models it serves and merges one provider into the settings (other providers and settings are left alone,
a `.bak` is written, and the key is never saved in the file: dsh reads it from `COLAB_LLM_API_KEY`). A quick-tunnel
URL changes every session; a Cloudflare named tunnel (see the notebook's notes) gives a fixed one.

## What it was checked against

- **Flags.** Every llama.cpp flag the hub passes was checked against `llama-server --help` of the commit the
  reference setup runs (`4c3c5810`, built CPU-only here), including `--reasoning`, `--chat-template-kwargs`,
  `--cache-ram`, `--ctx-checkpoints`.
- **Tests.** `python -m unittest discover -s colab_llm_hub/tests -t .` runs 41 tests: auth, model filtering,
  on-demand loading, switching, waiting for in-flight streams, keep-alives, slow loads, failed starts, crash
  recovery, prefetch, Drive save/restore, interrupted copies, eviction, resume of partial downloads, and merging
  the provider into dsh settings.
- **Real engine, no GPU.** The real hub, real Hugging Face downloader and the real llama.cpp binary served a
  135M model (non-streaming and streaming), saved it to a fake Drive, and a second "session" with the local disk
  wiped restored it from Drive without downloading.
- **Model repos.** Every repo and file name in `catalog.json` was confirmed to exist through the Hugging Face API;
  sizes are the real file sizes.

## What has **not** been validated

- Nothing here has run on a Colab GPU. No speed, memory or context numbers have been measured.
- Context sizes in `catalog.json` are conservative estimates for each amount of VRAM, not validated allocations.
- vLLM support for these model architectures on whichever vLLM pip installs, and on Blackwell (G4), is unverified.
  The Nemotron entry is marked experimental. FP8 on an A100 runs through a weight-only path (no native FP8).
- Colab's runtimes (A100 memory size, whether a G4 is the 96 GB RTX PRO 6000) are taken from public information and
  measured at run time; the catalog adapts to the reported VRAM.

## Compared with the local reference setup

| | Reference (2 x RX 7900 XTX, 48 GB total) | This hub on Colab |
|---|---|---|
| Large MoE models | CPU expert offload + GPU hot-expert cache | whole model in one GPU's VRAM (L4 24 GB, A100 40/80 GB, G4 ~96 GB) |
| Ornith-1.5-122B Q4_K_M (75.8 GB) | 11-13 decode tok/s with 30 CPU expert layers (their measurement) | fits entirely on a G4; speed not measured |
| Engines | llama.cpp, SGLang, vLLM, Strata, Maya | llama.cpp, vLLM |
| Not portable | Strata and Maya are AMD-specific; DeepSeek-V4.1-Flash (510 GB FP8) is far too large for one Colab GPU | |
| Switching/guards | model hub, GPU lock, VRAM/thermal guards | switching and draining only |
| Speculative decoding (MTP, DFlash) | used per profile | not enabled by default; add flags via `extra_args` in the catalog |

## Adding a model

Add an entry to `catalog.json` (the tests validate the file). For llama.cpp give the Hugging Face repo, the
`files` pattern to download, the `model_file` to load (for split GGUFs, the first shard), the size and the
`contexts` table `[[min VRAM GB, tokens], ...]`. For vLLM give the repo, parsers and `language_model_only`.
Extra engine flags go in `extra_args`.
