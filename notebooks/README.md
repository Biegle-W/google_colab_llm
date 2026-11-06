# Colab notebooks by engine

| Folder | Hub notebook | Other |
|---|---|---|
| _(root)_ | [`llm_hub_colab.ipynb`](llm_hub_colab.ipynb) index | — |
| [`llama/`](llama/) | [`llama_hub_colab.ipynb`](llama/llama_hub_colab.ipynb) | `llama_cpp_gpu.ipynb`, `llama_cpp_t4.ipynb` |
| [`vllm/`](vllm/) | [`vllm_hub_colab.ipynb`](vllm/vllm_hub_colab.ipynb) | `vllm_gpu.ipynb`, `vllm_tpu.ipynb` |
| [`sglang/`](sglang/) | [`sglang_hub_colab.ipynb`](sglang/sglang_hub_colab.ipynb) | — |
| [`diffusers/`](diffusers/) | [`diffusers_hub_colab.ipynb`](diffusers/diffusers_hub_colab.ipynb) | — |
| [`strata/`](strata/) | [`strata_hub_colab.ipynb`](strata/strata_hub_colab.ipynb) | — |
| [`maya/`](maya/) | [`maya_hub_colab.ipynb`](maya/maya_hub_colab.ipynb) | — |
| [`tensorfold/`](tensorfold/) | [`tensorfold_hub_colab.ipynb`](tensorfold/tensorfold_hub_colab.ipynb) | — |
| [`comfyui/`](comfyui/) | [`comfyui_hub_colab.ipynb`](comfyui/comfyui_hub_colab.ipynb) | — |
| [`sparsa/`](sparsa/) | [`sparsa_hub_colab.ipynb`](sparsa/sparsa_hub_colab.ipynb) | — |
| [`unsloth/`](unsloth/) | — | `Unsloth_Studio_Colab.ipynb` |

Regenerate hub notebooks from the repo root:

```bash
python3 -m colab_llm_hub.make_engine_notebooks
```
