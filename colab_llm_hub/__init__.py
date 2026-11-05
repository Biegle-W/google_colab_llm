"""OpenAI-compatible multi-model inference hub for Google Colab GPUs.

A hub process authenticates requests, starts or switches llama.cpp / vLLM / Diffusers on demand for
the model a request names, and keeps model files on Google Drive so they are downloaded only once.
Supports chat completions and text-to-image generation on Colab Pro / Google AI Pro GPUs.
"""
__version__ = "0.3.0"
