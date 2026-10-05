# Google Colab LLM notebooks (T4)

Notebooks for running LLM tooling on a free Colab **T4** GPU. Set Runtime → Change runtime type → **T4 GPU**, then Run all.

| Notebook | What it does |
|---|---|
| `studio/Unsloth_Studio_Colab.ipynb` | Official Unsloth Studio (web UI for training and chat) |
| `llama_cpp/llama_cpp_t4.ipynb` | Build llama.cpp with CUDA, download a GGUF model, chat via llama-server |

## Credit and license
The Studio notebook is from [unslothai/unsloth](https://github.com/unslothai/unsloth); Unsloth Studio is licensed under AGPL-3.0.
llama.cpp is from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
