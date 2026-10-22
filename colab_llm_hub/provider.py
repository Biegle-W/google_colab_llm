"""Settings for the local harness (dsh): a provider block for this hub, in the same shape the reference setup uses."""
import json


def _q(value):
    return json.dumps(value)          # JSON strings are valid YAML scalars


def provider_dict(base_url, models, api_key_env="COLAB_LLM_API_KEY", max_tokens=4096):
    """The provider block as data (same shape as the reference setup's local providers)."""
    return {"displayName": "Colab hub", "apiKeyEnv": api_key_env, "api": "openai-completions",
            "baseURL": base_url.rstrip("/") + "/v1", "timeoutMs": 86400000, "streamIdleTimeoutMs": 86400000,
            "compat": {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
            "models": [{"id": m["id"], "name": m["name"], "contextWindow": int(m["context_length"]),
                        "maxTokens": min(max_tokens, int(m["context_length"]) // 2),
                        "input": ["text"], "reasoningEfforts": False} for m in models]}


def merge_into_settings(doc, provider_id, provider, default_model=None):
    """Add/replace one provider in a dsh settings document without touching anything else.

    Handles both layouts the reference setup uses: a plain mapping (settings.yaml) and a list of
    {id, name, config} entries (the web profile patch).
    """
    if isinstance(doc, list):
        by_id = {e.get("id"): e for e in doc if isinstance(e, dict)}
        llm = by_id.get("llm-pi-ai")
        if llm is None:
            llm = {"id": "llm-pi-ai", "name": "@deepseek-ai/dsh-llm-pi-ai", "config": {}}
            doc.append(llm)
        llm.setdefault("config", {}).setdefault("providers", {})[provider_id] = provider
        if default_model:
            sel = by_id.get("agent-default-model")
            if sel is None:
                sel = {"id": "agent-default-model", "name": "@deepseek-ai/dsh-agent-default-model", "config": {}}
                doc.append(sel)
            sel["config"] = {"provider": provider_id, "model": default_model}
        return doc
    doc = doc if isinstance(doc, dict) else {}
    doc.setdefault("llm-pi-ai", {}).setdefault("providers", {})[provider_id] = provider
    if default_model:
        doc["agent-default-model"] = {"provider": provider_id, "model": default_model}
    return doc


def dsh_provider_yaml(base_url, models, provider_id="colab-hub", api_key_env="COLAB_LLM_API_KEY", max_tokens=4096):
    """`models` are the hub's /v1/models rows. Returns YAML to merge into dsh's settings under llm-pi-ai.providers."""
    lines = ["llm-pi-ai:", "  providers:", f"    {provider_id}:",
             "      displayName: Colab hub", f"      apiKeyEnv: {api_key_env}", "      api: openai-completions",
             f"      baseURL: {_q(base_url.rstrip('/') + '/v1')}",
             "      timeoutMs: 86400000", "      streamIdleTimeoutMs: 86400000",
             "      compat:", "        supportsDeveloperRole: false", "        maxTokensField: max_tokens",
             "      models:"]
    for m in models:
        lines += [f"        - id: {_q(m['id'])}", f"          name: {_q(m['name'])}",
                  f"          contextWindow: {int(m['context_length'])}",
                  f"          maxTokens: {min(max_tokens, int(m['context_length']) // 2)}",
                  "          input: [text]", "          reasoningEfforts: false"]
    return "\n".join(lines) + "\n"


def openai_env(base_url, api_key):
    """Shell lines for any OpenAI-compatible client."""
    return f'export OPENAI_BASE_URL="{base_url.rstrip("/")}/v1"\nexport OPENAI_API_KEY="{api_key}"\n'
