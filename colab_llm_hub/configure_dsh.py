"""Run on your own computer: add the Colab hub to dsh's settings.

    python -m colab_llm_hub.configure_dsh --url https://xxxx.trycloudflare.com --key <api key> \
        --settings path/to/settings.yaml [--default qwen3.8-27b-q4]

It asks the hub which models it serves (so context sizes match the Colab GPU) and merges one provider into the
settings file, leaving every other provider and setting alone. Needs PyYAML (pip install pyyaml); the key itself is
never written to the file: dsh reads it from the COLAB_LLM_API_KEY environment variable.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request

from . import provider


def fetch_models(url, key):
    req = urllib.request.Request(url.rstrip("/") + "/v1/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]


def configure(url, key, settings_path, provider_id="colab-hub", default=None, fetch=fetch_models):
    import yaml
    models = fetch(url, key)
    if default and default not in {m["id"] for m in models}:
        raise SystemExit(f"{default} is not served by this hub; it serves: {', '.join(m['id'] for m in models)}")
    doc = yaml.safe_load(open(settings_path).read()) if os.path.exists(settings_path) else {}
    if os.path.exists(settings_path):
        shutil.copyfile(settings_path, settings_path + ".bak")
    doc = provider.merge_into_settings(doc or {}, provider_id, provider.provider_dict(url, models), default)
    os.makedirs(os.path.dirname(os.path.abspath(settings_path)), exist_ok=True)
    tmp = settings_path + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    os.replace(tmp, settings_path)
    return models


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="the hub's public URL (without /v1)")
    p.add_argument("--key", default=os.environ.get("COLAB_LLM_API_KEY"), help="API key (default: $COLAB_LLM_API_KEY)")
    p.add_argument("--settings", required=True, help="dsh settings.yaml (or the web profile patch file)")
    p.add_argument("--provider-id", default="colab-hub")
    p.add_argument("--default", help="also make this model dsh's default")
    a = p.parse_args(argv)
    if not a.key:
        sys.exit("Give the API key with --key or the COLAB_LLM_API_KEY environment variable.")
    models = configure(a.url, a.key, a.settings, a.provider_id, a.default)
    print(f"Added provider {a.provider_id!r} with {len(models)} models to {a.settings} (backup: {a.settings}.bak).")
    print("Set COLAB_LLM_API_KEY in the environment dsh runs in, then restart dsh.")


if __name__ == "__main__":
    main()
