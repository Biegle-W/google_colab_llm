"""Move each engine pin in notebook_code/engine_versions.json to its newest compatible release.

No GPU and no Colab runtime: a candidate version is accepted only when
  - it is a stable release (the engine's `track` pattern) at least --min-age-days old,
  - every command-line flag our code passes to that engine is still in its source (flags),
  - for ComfyUI, every node class our workflows use (and each CLIPLoader type) still exists (nodes),
  - for Diffusers, every pipeline class the catalog names still exists (pipelines),
  - the release files the installer downloads exist (assets), and PyPI has wheels for the runtime's Python (wheels),
  - and, with the new pin, the notebooks regenerate and the unit tests pass.
A rejected candidate leaves the pin where it was and is reported. A pin that moves is marked as not yet GPU-tested
(the README's engine table says so) until a GPU run records it as `tested`.

    python3 tools/engine_updates.py              # report what would change
    python3 tools/engine_updates.py --apply      # apply, regenerate notebooks and README, run the tests
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "notebook_code" / "engine_versions.json"
WORKFLOWS = ROOT / "notebook_code" / "workflows"
sys.path.insert(0, str(ROOT))

# How each engine's version is stored and checked. store: sha (a commit), tag, or bare (the tag without its "v").
ENGINES = {
    "llama": {"store": "sha", "checks": ["flags"]},
    "comfyui": {"store": "tag", "checks": ["nodes"]},
    "tensorfold": {"store": "bare", "checks": ["assets", "flags"],
                   "assets": ["tensorfold-{bare}-linux-x86_64.tar.gz"]},
    "hyperqwen": {"store": "sha", "checks": ["flags"]},
    "ninfer": {"store": "sha", "checks": ["flags"]},
    "basalt": {"store": "sha", "checks": ["flags"]},
    "strata_nvfp4": {"store": "sha", "checks": ["flags"]},
    "strata": {"store": "tag", "checks": ["assets", "flags"], "assets": ["install.sh", "install-manifest.txt"]},
    "maya": {"store": "tag", "checks": ["flags"]},
    "vllm": {"store": "pypi", "checks": ["wheels", "flags"], "source": ("vllm-project/vllm", "v{version}")},
    "sglang": {"store": "pypi", "checks": ["wheels", "flags"], "source": ("sgl-project/sglang", "v{version}")},
    "diffusers": {"store": "pypi", "checks": ["wheels", "pipelines"], "source": ("huggingface/diffusers", "v{version}")},
}
STABLE_PYPI = re.compile(r"^\d+(\.\d+)*$")
# Files (and, after ":", a string in them) the installer's build steps rely on.
BUILD_FILES = {
    "llama": ["CMakeLists.txt", "tools/server/CMakeLists.txt:llama-server"],
    "comfyui": ["main.py", "requirements.txt"],
    "hyperqwen": ["docker/requirements.txt:vllm==", "patches/apply.sh"],
    "ninfer": ["CMakePresets.json:rtx5090-release"],
    "basalt": ["CMakeLists.txt", "requirements.txt"],
    "strata_nvfp4": ["setup.sh"],
    "maya": ["setup.sh", "maya.sh", "maya.py"],
}


def sh(cmd, cwd=None, timeout=900):
    return subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True,
                          timeout=timeout, check=True).stdout


def vkey(tag):
    return tuple(int(x) for x in re.findall(r"\d+", tag))


def remote_refs(repo):
    """{tag: commit} (annotated tags resolved to their commit) and the default branch's HEAD."""
    out = sh(["git", "ls-remote", f"https://github.com/{repo}.git", "HEAD", "refs/tags/*"], timeout=120)
    tags, head = {}, None
    for line in out.splitlines():
        sha, ref = line.split("\t")
        if ref == "HEAD":
            head = sha
        elif ref.startswith("refs/tags/"):
            name = ref[len("refs/tags/"):]
            if name.endswith("^{}"):
                tags[name[:-3]] = sha
            else:
                tags.setdefault(name, sha)
    return tags, head


def fetch_source(repo, ref, dest):
    """Shallow checkout of `ref` (tag or commit); returns its commit time (epoch seconds)."""
    url = f"https://github.com/{repo}.git"
    for cmd in (["git", "init", "-q", str(dest)], ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", url, ref],
                ["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"]):
        sh(cmd, timeout=1800)
    return int(sh(["git", "-C", str(dest), "log", "-1", "--format=%ct"]).strip())


def commit_times(repo, ref, depth=1):
    """[(commit, epoch)] newest first for `ref`, fetching commits only (no file contents)."""
    with tempfile.TemporaryDirectory() as d:
        sh(["git", "init", "-q", d])
        sh(["git", "-C", d, "fetch", "-q", "--depth", str(depth), "--filter=blob:none",
            f"https://github.com/{repo}.git", ref], timeout=300)
        out = sh(["git", "-C", d, "log", f"-{depth}", "--format=%H %ct", "FETCH_HEAD"])
    return [(c, int(t)) for c, t in (line.split() for line in out.splitlines())]


def http_ok(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "engine-updates"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status < 400
    except urllib.error.HTTPError:
        return False


def pypi(package):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=60) as r:
        return json.load(r)


def wheel_fits(files, python):
    """A wheel this Python on Linux x86_64 can install."""
    cp = "cp" + python.replace(".", "")
    for f in files:
        n = f["filename"]
        if not n.endswith(".whl") or f.get("yanked"):
            continue
        if n.endswith("-none-any.whl") or ((cp in n or "abi3" in n) and "x86_64" in n and "linux" in n):
            return True
    return False


def used_flags(engine):
    """Every --flag our command builders pass to this engine, over all its catalog entries."""
    from notebook_code import backends, catalog
    bins = {"llama": "llama-server", "python": "python", "builds": "/b", "work_dir": "/w"}
    flags = set()
    for e in catalog.load():
        if e["backend"] != engine:
            continue
        ctx = e["contexts"][-1][1]
        try:
            cmd = backends.command(e, "/models/x", ctx, bins)
        except Exception:                                  # an entry the builder cannot render offline
            continue
        for tok in " ".join(cmd).replace("'", " ").split():
            if tok.startswith("--") and len(tok) > 2:
                flags.add(tok.split("=", 1)[0])
    return sorted(flags)


def missing_strings(src, needles):
    """Needles (flags, class names) that appear nowhere in the source tree's text files."""
    left = set(needles)
    for path in Path(src).rglob("*"):
        if not left:
            break
        if ".git" in path.parts or not path.is_file() or path.stat().st_size > 4_000_000:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        left = {n for n in left if n not in text}
    return sorted(left)


def workflow_needs():
    nodes, clip_types = set(), set()
    for wf in WORKFLOWS.glob("*.api.json"):
        for node in json.loads(wf.read_text()).values():
            if isinstance(node, dict) and node.get("class_type"):
                nodes.add(node["class_type"])
                if node["class_type"] == "CLIPLoader" and node.get("inputs", {}).get("type"):
                    clip_types.add(node["inputs"]["type"])
    return sorted(nodes), sorted(clip_types)


def candidate(name, v, now, min_age):
    """(version to store, label, commit time or None) of the newest acceptable release, or None."""
    cfg = ENGINES[name]
    if cfg["store"] == "pypi":
        data = pypi(v["package"])
        best = None
        for ver, files in data["releases"].items():
            if not STABLE_PYPI.match(ver) or not files or not wheel_fits(files, v.get("python", "3.13")):
                continue
            uploaded = min(datetime.datetime.fromisoformat(f["upload_time_iso_8601"].replace("Z", "+00:00"))
                           for f in files).timestamp()
            if now - uploaded < min_age:
                continue
            if best is None or vkey(ver) > vkey(best[0]):
                best = (ver, uploaded)
        return (best[0], best[0], best[1]) if best else None
    tags, head = remote_refs(v["repo"])
    if v["track"] == "head":                   # no releases: the newest default-branch commit old enough
        for commit, when in commit_times(v["repo"], "HEAD", depth=100):
            if now - when >= min_age:
                return commit, commit[:8], when
        return None
    pattern = re.compile(v["track"].split(":", 1)[1])
    stable = sorted((t for t in tags if pattern.match(t)), key=vkey, reverse=True)
    for tag in stable[:60]:                    # the newest release old enough (llama.cpp tags every few hours)
        when = commit_times(v["repo"], f"refs/tags/{tag}")[0][1]
        if now - when >= min_age and not missing_assets(name, v, tag):   # a tag with no release files is skipped
            return {"sha": tags[tag], "tag": tag, "bare": tag.lstrip("v")}[cfg["store"]], tag, when
    return None


def source_ref(name, v, version, label):
    cfg = ENGINES[name]
    if cfg["store"] == "pypi":
        return cfg["source"][0], cfg["source"][1].format(version=version)
    return v["repo"], (label if cfg["store"] in ("tag", "bare") else version)


def check(name, v, version, label, work):
    """Problems with this candidate (empty = compatible) and its commit time."""
    cfg, problems = ENGINES[name], []
    repo, ref = source_ref(name, v, version, label)
    src = Path(work) / name
    try:
        when = fetch_source(repo, ref, src)
    except subprocess.CalledProcessError as e:
        return [f"could not fetch {repo} {ref}: {(e.stderr or '').strip()[:200]}"], None
    if "flags" in cfg["checks"]:
        gone = missing_strings(src, used_flags(name))
        if gone and v.get("version"):
            # Flags our own wrapper scripts take are in no engine release: only a flag the pinned version has and
            # the candidate lost is a break.
            try:
                cur = Path(work) / f"{name}-current"
                fetch_source(*source_ref(name, v, v["version"], v.get("label") or v["version"]), cur)
                gone = [f for f in gone if f not in missing_strings(cur, gone)]
            except subprocess.CalledProcessError:
                pass                                   # the pinned source is gone: judge the candidate alone
        if gone:
            problems.append("flags no longer in its source: " + ", ".join(gone))
    for spec in BUILD_FILES.get(name, []):
        rel, _, needle = spec.partition(":")
        f = src / rel
        if not f.is_file() or (needle and needle not in f.read_text(errors="ignore")):
            problems.append(f"build file missing or changed: {spec}")
    if "nodes" in cfg["checks"]:
        nodes, clip_types = workflow_needs()
        gone = missing_strings(src, [f'"{n}"' for n in nodes] + [f'"{t}"' for t in clip_types])
        if gone:
            problems.append("ComfyUI nodes / CLIP types missing: " + ", ".join(gone))
    if "pipelines" in cfg["checks"]:
        from notebook_code import catalog
        pipes = sorted({e["pipeline"] for e in catalog.load() if e.get("pipeline")})
        gone = missing_strings(src / "src", [f"class {p}" for p in pipes])
        if gone:
            problems.append("pipelines missing: " + ", ".join(gone))
    problems += [f"release file missing: {u}" for u in missing_assets(name, v, label)]
    return problems, when


def missing_assets(name, v, tag):
    cfg = ENGINES[name]
    if "assets" not in cfg["checks"]:
        return []
    urls = [f"https://github.com/{v['repo']}/releases/download/{tag}/{a.format(bare=tag.lstrip('v'))}"
            for a in cfg["assets"]]
    return [u for u in urls if not http_ok(u)]


def is_older(v, label):
    """True when a candidate release is older than the pin (a pin set by hand ahead of the two-day wait, e.g. for a
    model that needs that release): the updater never moves a pin backwards."""
    current = v.get("label") or ""
    if not v.get("version") or v.get("track") in ("head",) or not re.search(r"\d", current):
        return False
    try:
        return vkey(label) < vkey(current)
    except ValueError:
        return False


def regenerate_and_test():
    """Notebooks, README tables, notebook check and unit tests with the current pins; (ok, tail of the output)."""
    steps = [[sys.executable, "-m", "notebook_code.make_engine_notebooks"],
             [sys.executable, "-m", "notebook_code.results_table"],
             [sys.executable, "tools/check_notebooks.py"],
             [sys.executable, "-m", "unittest", "notebook_code.tests.test_units", "notebook_code.tests.test_hub"]]
    for cmd in steps:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            return False, (r.stdout + r.stderr)[-1500:]
    return True, ""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--apply", action="store_true", help="write the new pins (default: report only)")
    p.add_argument("--only", default="", help="comma-separated engines to consider")
    p.add_argument("--min-age-days", type=float, default=2.0,
                   help="skip releases newer than this (a broken release is usually replaced within a day or two)")
    a = p.parse_args(argv)
    versions = json.loads(VERSIONS.read_text())
    today = datetime.date.today().isoformat()
    now, min_age = time.time(), a.min_age_days * 86400
    report = []
    names = [n for n in ENGINES if not a.only or n in a.only.split(",")]
    with tempfile.TemporaryDirectory() as work:
        for name in names:
            v = versions[name]
            if v.get("track") in (None, "none"):
                continue
            try:
                cand = candidate(name, v, now, min_age)
            except Exception as e:
                report.append((name, "error", f"could not list releases: {e}"))
                continue
            if not cand or cand[0] == v.get("version") or is_older(v, cand[1]):
                report.append((name, "current", v.get("label") or v.get("version")))
                continue
            version, label, _ = cand
            problems, when = check(name, v, version, label, work)
            if not problems and when is not None and now - when < min_age:
                problems = [f"{label} is less than {a.min_age_days:g} days old; waiting"]
            if problems:
                report.append((name, "kept", f"{v.get('label')} (not {label}: {'; '.join(problems)})"))
                continue
            if not a.apply:
                report.append((name, "would update", f"{v.get('label')} -> {label}"))
                continue
            old = dict(v)
            v.update(version=version, label=label, updated=today)
            VERSIONS.write_text(json.dumps(versions, indent=2) + "\n")
            ok, tail = regenerate_and_test()
            if ok:
                report.append((name, "updated", f"{old.get('label')} -> {label}"))
            else:
                versions[name] = old
                VERSIONS.write_text(json.dumps(versions, indent=2) + "\n")
                report.append((name, "kept", f"{old.get('label')} (not {label}: the unit tests failed with it)\n"
                                             + tail))
    if a.apply:
        regenerate_and_test()                              # leave the notebooks matching the final pins
    for name, state, detail in report:
        print(f"{name:13} {state:13} {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
