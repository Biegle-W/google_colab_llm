# ComfyUI workflows for the Colab hub

- `*.api.json` — prompt graphs used by `comfyui_server` for OpenAI `/v1/images/generations`
  (stock ComfyUI loaders: `UNETLoader` / `CLIPLoader` / `VAELoader`).
- `*.ui.json` — workstation UI graphs (may reference custom `NvidiaResident*` nodes). Useful
  after you install matching custom nodes; not required for the OpenAI image path.
