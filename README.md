# Unsloth on Google Colab (T4)

Fine-tune LLMs with [Unsloth](https://github.com/unslothai/unsloth) on a free/Pro Colab **T4** GPU.

## Quick start
1. Open `unsloth_t4_finetune.ipynb` in Colab (File → Open notebook → GitHub, or upload it).
2. Runtime → Change runtime type → **T4 GPU**. (Unsloth needs CUDA; it does not run on TPU.)
3. Run all cells.

## T4 limits
- 16 GB VRAM, fp16 only (no bf16).
- Use 4-bit models of 8B or smaller; 3B-7B is comfortable.
- If you hit out-of-memory errors, lower `MAX_SEQ_LEN` and `BATCH_SIZE` in the config cell.
- Colab disk is temporary: save LoRA adapters to Google Drive or the Hugging Face Hub.

## Files
- `unsloth_t4_finetune.ipynb` - install, load, LoRA, train, infer, save.
- `requirements.txt` - for running outside Colab on a CUDA machine.
