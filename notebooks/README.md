# Colab notebooks by engine

Each engine has one GPU notebook (L4, A100 or G4). It runs the hub: one authenticated OpenAI-compatible
endpoint for the model you pick, with model files kept on Google Drive. The T4 and TPU have their own
single-server notebooks, because no hub model fits a T4 and the hub does not run on TPUs.

| Folder | GPU notebook | Other |
|---|---|---|
| _(root)_ | [`llm_hub_colab.ipynb`](llm_hub_colab.ipynb) index | — |
| [`llama/`](llama/) | [`llama_gpu.ipynb`](llama/llama_gpu.ipynb) | [`llama_cpp_t4.ipynb`](llama/llama_cpp_t4.ipynb) (T4) |
| [`vllm/`](vllm/) | [`vllm_gpu.ipynb`](vllm/vllm_gpu.ipynb) | [`vllm_tpu.ipynb`](vllm/vllm_tpu.ipynb) (TPU) |
| [`sglang/`](sglang/) | [`sglang_gpu.ipynb`](sglang/sglang_gpu.ipynb) | — |
| [`diffusers/`](diffusers/) | [`diffusers_gpu.ipynb`](diffusers/diffusers_gpu.ipynb) | — |
| [`strata/`](strata/) | [`strata_gpu.ipynb`](strata/strata_gpu.ipynb) | — |
| [`maya/`](maya/) | [`maya_gpu.ipynb`](maya/maya_gpu.ipynb) | — |
| [`tensorfold/`](tensorfold/) | [`tensorfold_gpu.ipynb`](tensorfold/tensorfold_gpu.ipynb) | — |
| [`comfyui/`](comfyui/) | [`comfyui_gpu.ipynb`](comfyui/comfyui_gpu.ipynb) | — |
| [`sparsa/`](sparsa/) | [`sparsa_gpu.ipynb`](sparsa/sparsa_gpu.ipynb) | — |
| [`hyperqwen/`](hyperqwen/) | [`hyperqwen_gpu.ipynb`](hyperqwen/hyperqwen_gpu.ipynb) | A100: batch entry only; L4 too small |
| [`ninfer/`](ninfer/) | [`ninfer_gpu.ipynb`](ninfer/ninfer_gpu.ipynb) | G4 (sm_120) only |
| [`unsloth/`](unsloth/) | — | `Unsloth_Studio_Colab.ipynb` |

Regenerate the GPU notebooks from the repo root:

```bash
python3 -m colab_llm_hub.make_engine_notebooks
```
