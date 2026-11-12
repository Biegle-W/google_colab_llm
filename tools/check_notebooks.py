"""Refuse to publish notebooks that carry outputs: an executed cell can hold API keys, tunnel URLs or tokens.

python tools/check_notebooks.py [paths...]   (default: every .ipynb in the repo). Exit 1 lists the offenders.
Clean them by regenerating (python -m notebook_code.make_engine_notebooks) or with "Edit > Clear all outputs".
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def problems(path):
    nb = json.loads(pathlib.Path(path).read_text())
    found = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("outputs"):
            found.append(f"cell {i}: has outputs")
        if cell.get("execution_count") is not None:
            found.append(f"cell {i}: execution_count {cell['execution_count']}")
    return found


def main(paths):
    files = [pathlib.Path(p) for p in paths] or sorted(ROOT.rglob("*.ipynb"))
    bad = {}
    for f in files:
        if ".ipynb_checkpoints" in f.parts or "node_modules" in f.parts:
            continue
        if found := problems(f):
            bad[f] = found
    for f, found in bad.items():
        name = f.resolve().relative_to(ROOT) if f.resolve().is_relative_to(ROOT) else f
        print(f"{name}: " + "; ".join(found[:3]))
    if bad:
        print(f"{len(bad)} notebook(s) carry outputs; clear them before publishing (they can contain API keys).")
        return 1
    print(f"OK: {len(files)} notebooks, no outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
