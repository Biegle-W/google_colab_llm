"""Settings for the local harness (dsh): a provider block for this hub, in the same shape the reference setup uses."""
import json


def _q(value):
    return json.dumps(value)          # JSON strings are valid YAML scalars


def _chat_models(models):
    """dsh only speaks chat completions; skip Diffusers / image catalog rows."""
    return [m for m in models if m.get("kind", "chat") != "image" and m.get("backend") != "diffusers"]


def provider_dict(base_url, models, api_key_env="COLAB_LLM_API_KEY", max_tokens=4096):
    """The provider block as data (same shape as the reference setup's local providers)."""
    return {"displayName": "Colab hub", "apiKeyEnv": api_key_env, "api": "openai-completions",
            "baseURL": base_url.rstrip("/") + "/v1", "timeoutMs": 86400000, "streamIdleTimeoutMs": 86400000,
            "compat": {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
            "models": [{"id": m["id"], "name": m["name"], "contextWindow": int(m["context_length"]),
                        "maxTokens": min(max_tokens, int(m["context_length"]) // 2),
                        "input": ["text"], "reasoningEfforts": False} for m in _chat_models(models)]}


def dsh_provider_yaml(base_url, models, provider_id="colab-hub", api_key_env="COLAB_LLM_API_KEY", max_tokens=4096):
    """`models` are the hub's /v1/models rows. Returns YAML to merge into dsh's settings under llm-pi-ai.providers."""
    lines = ["llm-pi-ai:", "  providers:", f"    {provider_id}:",
             "      displayName: Colab hub", f"      apiKeyEnv: {api_key_env}", "      api: openai-completions",
             f"      baseURL: {_q(base_url.rstrip('/') + '/v1')}",
             "      timeoutMs: 86400000", "      streamIdleTimeoutMs: 86400000",
             "      compat:", "        supportsDeveloperRole: false", "        maxTokensField: max_tokens",
             "      models:"]
    for m in _chat_models(models):
        lines += [f"        - id: {_q(m['id'])}", f"          name: {_q(m['name'])}",
                  f"          contextWindow: {int(m['context_length'])}",
                  f"          maxTokens: {min(max_tokens, int(m['context_length']) // 2)}",
                  "          input: [text]", "          reasoningEfforts: false"]
    return "\n".join(lines) + "\n"


def openai_env(base_url, api_key):
    """Shell lines for any OpenAI-compatible client."""
    return f'export OPENAI_BASE_URL="{base_url.rstrip("/")}/v1"\nexport OPENAI_API_KEY="{api_key}"\n'
