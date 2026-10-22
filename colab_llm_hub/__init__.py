"""OpenAI-compatible multi-model inference hub for Google Colab GPUs.

A hub process authenticates requests, starts or switches llama.cpp / vLLM on demand for the model
a request names, and keeps model files on Google Drive so they are downloaded only once.
"""
__version__ = "0.1.0"
