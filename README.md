# Google Colab LLM notebooks (T4)

Notebooks for running LLM tooling on a free Colab **T4** GPU. Set Runtime → Change runtime type → **T4 GPU**, then Run all.

| Notebook | What it does |
|---|---|
| `Unsloth_Studio_Colab.ipynb` | Official Unsloth Studio (web UI for training and chat) |
| `llama_cpp_t4.ipynb` | Build llama.cpp with CUDA, run Qwen3.8-27B Q4_K_M (partial GPU offload on a T4), chat via llama-server |
| `vllm_tpu.ipynb` | vLLM on a v5e-1 TPU; checks whether the model fits in 16 GB HBM and falls back to Qwen3-4B (the 27B does not fit) |

## Credit and license
The Studio notebook is from [unslothai/unsloth](https://github.com/unslothai/unsloth); Unsloth Studio is licensed under AGPL-3.0.
llama.cpp is from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).

## Model and build storage
All three notebooks mount Google Drive and keep models in `MyDrive/llm_models`, so a model is downloaded once and reused in later sessions (a 27B Q4 file needs about 17 GB of Drive space). The first run asks you to authorize Drive access.

## Remote access
`llama_cpp_t4.ipynb` and `vllm_tpu.ipynb` can open a free Cloudflare tunnel to their OpenAI-compatible server, protected by a random API key printed in the notebook. Both also run DeepSeek Harness (dsh) as an agent front end behind a tokenized link; llama.cpp additionally serves its own web UI. The URL and key last for one session.

Builds are cached too, in `MyDrive/llm_models/builds`: the Unsloth Studio install, the compiled llama.cpp binaries, and the vLLM packages. The first run builds and saves them; later runs restore them. Set the `REBUILD`, `REBUILD_LLAMA`, or `REFRESH_WHEELS` flag in the notebook to update to the latest versions. A cache is tied to the Python version (and, for llama.cpp, the T4), so if Colab changes its image the notebook rebuilds automatically.
