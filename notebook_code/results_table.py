"""Render results/results.csv (measured on Colab) as the README's "Tested on Colab" tables.

python -m notebook_code.results_table   rewrites the section between the RESULTS markers in README.md.
"""
import csv
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV = os.path.join(ROOT, "results", "results.csv")
README = os.path.join(ROOT, "README.md")
CODE_README = os.path.join(ROOT, "notebook_code", "README.md")
U_START, U_END = "<!-- UNTESTED:START -->", "<!-- UNTESTED:END -->"
E_START, E_END = "<!-- ENGINES:START -->", "<!-- ENGINES:END -->"
VERSIONS = os.path.join(ROOT, "notebook_code", "engine_versions.json")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
RUNTIMES = (("l4", "L4 (RTX 4070 Ti class, 24 GB)"), ("a100", "A100 80 GB (High-RAM)"),
            ("g4", "G4 (RTX PRO 6000 Blackwell, 96 GB)"))


def load(path=CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v, unit, fmt="{:,.0f}"):
    return (fmt.format(float(v)) + unit) if v not in ("", None) else "next run"


def _time(v):
    if v in ("", None):
        return "not recorded"
    s = float(v)
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.1f} min".replace(".0 min", " min")


def render(rows):
    out = [START, "", "## Tested on Colab", "",
           "Measured with the notebooks' hub on real Colab runtimes, one request at a time; the numbers live in",
           "[`results/results.csv`](results/results.csv) and the images and videos in",
           "[`results/samples/`](results/samples/). Sorted by runtime, then engine.", "",
           "- **Prefill:** prompt tokens per second for a ~3.5K-token prompt (fresh, no prefix-cache reuse).",
           "- **Time to first token:** for that prompt, on a loaded model.",
           "- **Decode:** a 512-token reply at temperature 0.",
           "- **Cold start:** the hub starting the engine, loading the model from Drive and answering a short prompt.",
           "- *next run*: measured from the next test run on (earlier runs used a prompt too short for prefill).", ""]
    from .recommend import notebook_plan
    planned = {(rt, m) for rt, _, m in notebook_plan()}
    dropped = [r for r in rows if (r["runtime"], r["model"]) not in planned]
    rows = [r for r in rows if (r["runtime"], r["model"]) in planned]
    for rt, title in RUNTIMES:
        mine = sorted((r for r in rows if r["runtime"] == rt), key=lambda r: (r["engine"].lower(), r["model"]))
        chat = [r for r in mine if r["kind"] == "chat"]
        media = [r for r in mine if r["kind"] != "chat"]
        if not mine:
            continue
        out += [f"### {title}", ""]
        if chat:
            out += ["| Engine | Model notebook | Prefill | Time to first token | Decode | Cold start |",
                    "|---|---|---|---|---|---|"]
            out += [f"| {r['engine']} | `{r['model']}` | {_num(r['prefill_tok_s'], ' tok/s')} | "
                    f"{_num(r['ttft_s'], ' s', '{:.2f}')} | {_num(r['decode_tok_s'], ' tok/s')} | "
                    f"{_time(r['cold_start_s'])} |" for r in chat]
            out += [""]
        if media:
            out += ["| Engine | Model notebook | Per image / 5 s clip (after the first) | Cold start | Sample |",
                    "|---|---|---|---|---|"]
            for r in media:
                per = f"{float(r['per_item_s']):.0f} s per {'clip' if r['kind'] == 'video' else 'image'}"
                if not r["sample"]:
                    sample = "not saved"
                elif r["sample"].endswith((".jpg", ".png", ".webp")):
                    sample = f'<a href="results/{r["sample"]}"><img src="results/{r["sample"]}" width="160"></a>'
                else:
                    sample = f"[{os.path.basename(r['sample'])}](results/{r['sample']})"
                out.append(f"| {r['engine']} | `{r['model']}` | {per} | {_time(r['cold_start_s'])} | {sample} |")
            out += [""]
    if dropped:
        out += ["Tested, then removed from the notebooks (a faster engine or quant, or released before June 2026; kept in the CSV): " + ", ".join(
            f"{r['runtime'].upper()} {r['engine']} `{r['model']}` ({_num(r['decode_tok_s'], ' tok/s')})"
            for r in dropped) + ".", ""]
    out += [END]
    return "\n".join(out)


def render_untested(rows):
    """Notebooks with no result in results.csv yet, per runtime."""
    from .recommend import notebook_plan
    done = {(r["runtime"], r["model"]) for r in rows}
    out = [U_START, "", "Notebooks not yet run on their GPU runtime (no row in `results/results.csv`):", ""]
    for rt, title in RUNTIMES:
        todo = sorted(f"`{e}/{m}`" for r, e, m in notebook_plan() if r == rt and (rt, m) not in done)
        out.append(f"- **{title}:** " + (", ".join(todo) if todo else "none, all tested") + ".")
    out += ["", U_END]
    return "\n".join(out)


def render_engines(versions=None):
    """Each engine's pinned version (notebook_code/engine_versions.json) and whether a GPU run has used it."""
    import json
    versions = versions if versions is not None else json.load(open(VERSIONS))
    out = [E_START, "", "Engine versions the notebooks install (`notebook_code/engine_versions.json`, moved forward "
           "nightly by `tools/engine_updates.py`; a new version is used from the next session on):", "",
           "| Engine | Pinned | Since | GPU-tested with |", "| --- | --- | --- | --- |"]
    for name, v in sorted(versions.items()):
        if v.get("track") == "none":
            continue
        tested = v.get("tested")
        if not v.get("version"):
            state = "unpinned (copy cached on Drive)"
        elif tested == v["version"]:
            state = "this version"
        elif tested:
            state = "an older version: **retest on GPU**"
        else:
            state = "not yet: **test on GPU**"
        out.append(f"| {name} | {v.get('label') or v.get('version')} | {v.get('updated', '')} | {state} |")
    out += ["", E_END]
    return "\n".join(out)


def _replace(path, start, end, block):
    text = open(path).read()
    a, b = text.index(start), text.index(end) + len(end)
    new = text[:a] + block + text[b:]
    open(path, "w").write(new)
    return new


def update_readme(path=README, rows=None):
    rows = rows if rows is not None else load()
    if os.path.isfile(CODE_README):
        _replace(CODE_README, U_START, U_END, render_untested(rows))
        if E_START in open(CODE_README).read():
            _replace(CODE_README, E_START, E_END, render_engines())
    return _replace(path, START, END, render(rows))


if __name__ == "__main__":
    update_readme()
    print("README.md and notebook_code/README.md updated from", os.path.relpath(CSV, ROOT))
