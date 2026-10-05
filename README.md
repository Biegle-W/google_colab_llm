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
