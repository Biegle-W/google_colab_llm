"""Model catalog: which models exist, which fit a given GPU, and with what context."""
import json
from pathlib import Path

BACKENDS = ("llama", "vllm")
REQUIRED = ("id", "name", "backend", "repo", "files", "size_gb", "contexts")
DEFAULT_PATH = Path(__file__).with_name("catalog.json")


class CatalogError(ValueError):
    pass


def validate(entries):
    seen = set()
    for e in entries:
        missing = [k for k in REQUIRED if k not in e]
        if missing:
            raise CatalogError(f"{e.get('id', '?')}: missing {missing}")
        if e["id"] in seen:
            raise CatalogError(f"duplicate model id {e['id']}")
        seen.add(e["id"])
        if e["backend"] not in BACKENDS:
            raise CatalogError(f"{e['id']}: unknown backend {e['backend']!r}")
        if e["backend"] == "llama" and "model_file" not in e:
            raise CatalogError(f"{e['id']}: llama entries need model_file")
        ctx = e["contexts"]
        if not ctx or any(len(c) != 2 for c in ctx):
            raise CatalogError(f"{e['id']}: contexts must be [[min_vram_gb, tokens], ...]")
        if [c[0] for c in ctx] != sorted(c[0] for c in ctx) or [c[1] for c in ctx] != sorted(c[1] for c in ctx):
            raise CatalogError(f"{e['id']}: contexts must increase in both columns")
    return entries


def load(path=None):
    return validate(json.loads(Path(path or DEFAULT_PATH).read_text()))


def min_vram_gb(entry):
    return entry["contexts"][0][0]


def fits(entry, vram_gb):
    return vram_gb >= min_vram_gb(entry)


def context_for(entry, vram_gb, override=None):
    """Largest listed context whose VRAM threshold the GPU meets (or the caller's override)."""
    if override:
        return int(override)
    if not fits(entry, vram_gb):
        raise CatalogError(f"{entry['id']} needs at least {min_vram_gb(entry)} GB of VRAM")
    best = entry["contexts"][0][1]
    for need, tokens in entry["contexts"]:
        if vram_gb >= need:
            best = tokens
    return best


def visible(entries, vram_gb):
    return [e for e in entries if fits(e, vram_gb)]


def by_id(entries):
    return {e["id"]: e for e in entries}


def for_engines(entries, engines):
    """Keep only models whose engine is installed (so the hub never advertises a model it cannot start)."""
    engines = set(engines)
    return [e for e in entries if e["backend"] in engines]
