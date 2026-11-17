"""Engine installs, cached on Google Drive so a new Colab session restores them instead of rebuilding.

llama.cpp is compiled for the detected GPU architecture and saved as a tarball of build/bin. vLLM / SGLang /
Diffusers wheels are saved in a wheelhouse on Drive and installed offline in later sessions. Install logs
also go under the Drive workspace ``logs/`` when ``COLAB_LLM_DRIVE`` is set.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import gpu

# Every engine's version lives in engine_versions.json (tools/engine_updates.py moves them forward). Drive caches are
# named after the version, so a new pin is picked up by the next session; None = the copy already cached on Drive.
VERSIONS = json.loads((Path(__file__).with_name("engine_versions.json")).read_text())


def pin(engine):
    return VERSIONS[engine]["version"]


LLAMA_URL = "https://github.com/ggml-org/llama.cpp"
LLAMA_PIN = pin("llama")     # fetched by SHA, so it stays reproducible
SRC = "/content/llama.cpp"


class InstallError(RuntimeError):
    pass


def _install_log_path():
    drive = os.environ.get("COLAB_LLM_DRIVE")
    if drive:
        path = os.path.join(drive, "logs", "install.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    return "/content/install.log"


def sh(cmd, log_path=None):
    log_path = log_path or _install_log_path()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    with open(log_path, "a") as f:
        f.write(f"$ {cmd}\n{r.stdout}{r.stderr}\n")
    if r.returncode:
        raise InstallError(f"Failed: {cmd}\n" + (r.stdout + r.stderr)[-3000:])
    return r.stdout


def llama_fetch_commands(ref, src=SRC):
    """Commands that put llama.cpp source for `ref` (a commit SHA, or 'latest') into `src`."""
    if ref in (None, "", "latest"):
        return [f"rm -rf {src} && git clone --depth 1 {LLAMA_URL} {src}"]
    return [f"rm -rf {src} && mkdir -p {src}", f"cd {src} && git init -q",
            f"cd {src} && git fetch -q --depth 1 {LLAMA_URL} {ref}", f"cd {src} && git checkout -q FETCH_HEAD"]


def llama_tarball(builds_dir, ref, arch_cmake):
    return os.path.join(builds_dir, f"llama.cpp_{(ref or 'latest')[:10]}_cuda_sm{arch_cmake}.tar")


def ensure_llama(builds_dir, arch, ref=LLAMA_PIN, rebuild=False, run=sh, log=print, src=SRC, check=True):
    """Return the path to llama-server for this GPU, restoring a cached build or building (and caching) one.

    check=False builds for `arch` without running the binary, for a CPU runtime preparing a GPU's build:
    nvcc does not need the GPU, only the CUDA toolkit."""
    os.environ["PATH"] += ":/usr/local/cuda/bin"
    cuda = gpu.nvcc_version(subprocess.run("nvcc --version", shell=True, capture_output=True, text=True).stdout)
    if cuda is None:
        raise InstallError("nvcc (the CUDA compiler) was not found on this runtime.")
    arch_cmake = gpu.cmake_arch(arch, cuda)
    if arch_cmake != str(arch):
        log(f"CUDA {cuda[0]}.{cuda[1]} cannot compile sm_{arch}; using {arch_cmake} (works, may be slightly slower).")
    os.makedirs(builds_dir, exist_ok=True)
    tarball, binary = llama_tarball(builds_dir, ref, arch_cmake), f"{src}/build/bin/llama-server"
    if not check and not rebuild and os.path.exists(tarball):
        log("llama.cpp build for sm_%s is already on Drive: %s" % (arch_cmake, tarball))
        return tarball

    def works():
        try:
            return subprocess.run([binary, "--version"], capture_output=True, timeout=120).returncode == 0
        except Exception:
            return False
    if not rebuild and os.path.exists(tarball):
        log("Restoring the cached llama.cpp build from Drive...")
        os.makedirs(f"{src}/build", exist_ok=True)
        run(f"tar -xf {tarball} -C {src}/build")
    if works():
        return binary
    log(f"Building llama.cpp ({(ref or 'latest')[:10]}) for sm_{arch_cmake}; several minutes, one time...")
    for cmd in llama_fetch_commands(ref, src):
        run(cmd)
    run(f"cd {src} && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch_cmake} -DLLAMA_CURL=OFF")
    run(f"cd {src} && cmake --build build --config Release -j $(nproc) --target llama-server")
    run(f"tar -cf {tarball}.part -C {src}/build bin && mv {tarball}.part {tarball}")
    log("Saved the build to Drive:", tarball)
    if not check:
        return tarball
    if not works():
        raise InstallError("llama-server was built but does not start; see /content/install.log")
    return binary


def _cache_wheels(wheels, pkgs, refresh, run, log):
    """Fill a Drive wheelhouse without installing (wheels are the same with or without a GPU)."""
    marker = os.path.join(wheels, "COMPLETE")
    if os.path.isfile(marker) and not refresh:
        log("Wheels already on Drive:", wheels)
        return wheels
    log(f"Downloading wheels for {pkgs} to Drive (one time)...")
    os.makedirs(wheels, exist_ok=True)
    run(f"{sys.executable} -m pip wheel -q -w {wheels} {pkgs}")
    open(marker, "w").write(pkgs)                      # an interrupted download is not marked complete
    return wheels


VLLM_VERSION = pin("vllm")
VLLM_SPEC = f"vllm=={VLLM_VERSION}" if VLLM_VERSION else "vllm"


def _versioned(name, version):
    """A Drive cache name for an engine version (the unversioned name is the copy cached before versions were pinned)."""
    return f"{name}-{version}" if version else name


def vllm_wheelhouse(builds_dir):
    return os.path.join(builds_dir, _versioned("vllm", VLLM_VERSION) + f"_gpu_py{sys.version_info.major}{sys.version_info.minor}")


def ensure_vllm(builds_dir, refresh=False, run=sh, log=print, install_now=True):
    """Install vLLM from wheels cached on Drive (downloading them once).

    install_now=False only fills the wheel cache (a CPU runtime preparing for a GPU one)."""
    wheels = vllm_wheelhouse(builds_dir)
    pip = f"{sys.executable} -m pip"
    if not install_now:
        return _cache_wheels(wheels, f"'{VLLM_SPEC}'", refresh, run, log)
    cached = bool(glob.glob(f"{wheels}/*.whl")) and not refresh
    if cached:
        log("Installing vLLM from the wheels saved on Drive...")
        try:
            run(f"{pip} install -q --no-index --find-links {wheels} '{VLLM_SPEC}'")
        except InstallError as e:
            log("Cached install failed, rebuilding the wheel cache:", str(e)[:300])
            cached = False
    if not cached:
        log("Downloading vLLM and its dependencies (several GB, one time)...")
        os.makedirs(wheels, exist_ok=True)
        run(f"{pip} wheel -q -w {wheels} '{VLLM_SPEC}'")
        run(f"{pip} install -q --no-index --find-links {wheels} '{VLLM_SPEC}'")
    return run(f"{pip} show vllm").split("Version:")[-1].split()[0]


# Diffusers stack used for text-to-image. Torch is already present on Colab GPU images; we cache the rest.
DIFFUSERS_VERSION = pin("diffusers")
DIFFUSERS_PKGS = (f"diffusers=={DIFFUSERS_VERSION}" if DIFFUSERS_VERSION else "diffusers", "transformers", "accelerate", "safetensors", "sentencepiece",
                  "protobuf", "Pillow", "bitsandbytes", "huggingface_hub", "hf_xet")


def diffusers_wheelhouse(builds_dir):
    return os.path.join(builds_dir, _versioned("diffusers", DIFFUSERS_VERSION) + f"_gpu_py{sys.version_info.major}{sys.version_info.minor}")


def ensure_diffusers(builds_dir, refresh=False, run=sh, log=print, install_now=True):
    """Install Diffusers (+ transformers/accelerate/bitsandbytes) from wheels cached on Drive."""
    wheels = diffusers_wheelhouse(builds_dir)
    pip = f"{sys.executable} -m pip"
    pkgs = " ".join(DIFFUSERS_PKGS)
    if not install_now:
        return _cache_wheels(wheels, pkgs, refresh, run, log)
    cached = bool(glob.glob(f"{wheels}/*.whl")) and not refresh
    if cached:
        log("Installing Diffusers from the wheels saved on Drive...")
        try:
            run(f"{pip} install -q --no-index --find-links {wheels} {pkgs}")
        except InstallError as e:
            log("Cached Diffusers install failed, rebuilding the wheel cache:", str(e)[:300])
            cached = False
    if not cached:
        log("Downloading Diffusers and its dependencies (one time; saved to Drive)...")
        os.makedirs(wheels, exist_ok=True)
        run(f"{pip} wheel -q -w {wheels} {pkgs}")
        run(f"{pip} install -q --no-index --find-links {wheels} {pkgs}")
    return run(f"{pip} show diffusers").split("Version:")[-1].split()[0]


def ensure_hub_deps(builds_dir, refresh=False, run=sh, log=print):
    """Small always-on deps (huggingface_hub) from Drive wheels — used before model prefetch."""
    wheels = os.path.join(builds_dir, f"hub_deps_py{sys.version_info.major}{sys.version_info.minor}")
    pip = f"{sys.executable} -m pip"
    pkgs = "huggingface_hub hf_xet"
    cached = bool(glob.glob(f"{wheels}/*.whl")) and not refresh
    if cached:
        log("Installing hub deps from Drive wheels...")
        try:
            run(f"{pip} install -q --no-index --find-links {wheels} {pkgs}")
            return
        except InstallError as e:
            log("Cached hub-deps install failed:", str(e)[:200])
            cached = False
    log("Caching huggingface_hub wheels on Drive...")
    os.makedirs(wheels, exist_ok=True)
    run(f"{pip} wheel -q -w {wheels} {pkgs}")
    run(f"{pip} install -q --no-index --find-links {wheels} {pkgs}")


def sglang_wheelhouse(builds_dir):
    return os.path.join(builds_dir, f"sglang_gpu_py{sys.version_info.major}{sys.version_info.minor}")


SGLANG_ROOT = "/content/sglang"
SGLANG_PYTHON = "3.12"      # sglang[all] needs outlines_core, which has no wheels for Colab's Python 3.13
SGLANG_VERSION = pin("sglang")
SGLANG_SPEC = f"sglang[all]=={SGLANG_VERSION}" if SGLANG_VERSION else "sglang[all]"


def sglang_tarball(builds_dir):
    return os.path.join(builds_dir, _versioned("sglang", SGLANG_VERSION) + f"_py{SGLANG_PYTHON.replace('.', '')}.tar")


def ensure_sglang(builds_dir, refresh=False, run=sh, log=print, install_now=True, root=SGLANG_ROOT):
    """SGLang in its own uv venv (Python 3.12, managed by uv inside `root`), cached on Drive as one tarball.

    Returns the venv's python (the hub runs `python -m sglang.launch_server` with it). Needs no GPU, so a CPU
    runtime can build the tarball (install_now=False returns once it is on Drive)."""
    tar = sglang_tarball(builds_dir)
    python = os.path.join(root, "venv", "bin", "python")
    if not install_now and os.path.isfile(tar) and not refresh:
        log("SGLang build already on Drive:", tar)
        return tar
    uv = f"UV_PYTHON_INSTALL_DIR={root}/.python UV_CACHE_DIR=/content/.uv-cache {sys.executable} -m uv"

    def with_ninja():
        # SGLang compiles some kernels at start-up for GPUs it ships none for (the A100's sm_80) and calls ninja.
        if not os.path.isfile(os.path.join(root, "venv", "bin", "ninja")):
            run(f"cd {root} && {uv} pip install --python venv/bin/python ninja")
        return python
    if os.path.isfile(python) and not refresh:
        return with_ninja()
    if os.path.isfile(tar) and not refresh:
        log("Restoring SGLang (its venv) from Drive:", tar)
        run(f"rm -rf {root} && mkdir -p {os.path.dirname(root)} && tar -xf {tar} -C {os.path.dirname(root)}")
        return with_ninja()
    log("Building SGLang's venv (Python 3.12, several GB, one time)...")
    run(f"rm -rf {root} && mkdir -p {root}")
    for cmd in _uv_venv_commands(root, "venv", python_version=SGLANG_PYTHON):
        run(cmd)
    run(f"cd {root} && {uv} pip install --python venv/bin/python '{SGLANG_SPEC}' ninja")
    version = run(f"{python} -c 'import sglang; print(sglang.__version__)'").strip().splitlines()[-1]
    log("SGLang", version)
    os.makedirs(builds_dir, exist_ok=True)
    run(f"tar -cf {tar}.tmp -C {os.path.dirname(root)} {os.path.basename(root)} && mv {tar}.tmp {tar}")
    log("Cached SGLang on Drive:", tar)
    return python if install_now else tar


STRATA_VERSION = pin("strata")
STRATA_FEED = (f"https://github.com/Niko1221/Strata/releases/download/{STRATA_VERSION}" if STRATA_VERSION
               else "https://github.com/Niko1221/Strata/releases/latest/download")
STRATA_INSTALL_SH = f"{STRATA_FEED}/install.sh"
MAYA_REPO = "https://github.com/mw00/project-maya.git"
MAYA_VERSION = pin("maya")


def local_tree(name, builds_dir, ready, build, refresh=False, run=sh, log=print, title=None, cache=None):
    """An engine folder that runs from local disk, cached on Drive as one tarball.

    Colab's Drive mount cannot hold executables or symlinks (it is mounted noexec and refuses ln -s), so an
    engine is never run from Drive: it is built in /content/<name>, saved as <builds>/<name>.tar, and later
    sessions unpack that tarball. `ready(root)` returns the path to hand back once the folder is usable."""
    title = title or name
    root = os.path.join("/content", name)
    tar = os.path.join(builds_dir, f"{cache or name}.tar")      # cache: a versioned tarball name for the same folder
    if not refresh:
        got = ready(root)
        if got:
            return got
        if os.path.isfile(tar):
            log(f"Restoring {title} from Drive:", tar)
            try:
                run(f"rm -rf {root} && tar -xf {tar} -C /content")
            except Exception as e:      # a tarball cut off mid-upload (runtime stopped before Drive synced)
                log(f"Could not unpack the cached {title} ({e})")
            got = ready(root)
            if got:
                return got
            log(f"The cached {title} is incomplete; installing it again")
    run(f"rm -rf {root} && mkdir -p {root}")
    build(root)
    got = ready(root)
    if not got:
        raise InstallError(f"{title} install finished without a usable result under {root}; see logs/install.log")
    os.makedirs(builds_dir, exist_ok=True)
    run(f"tar -cf {tar}.tmp -C /content {name} && mv {tar}.tmp {tar}")
    log(f"Cached {title} on Drive:", tar)
    return got


def _executable(*paths):
    return next((p for p in paths if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)


def ensure_strata(builds_dir, refresh=False, run=sh, log=print):
    """The Strata Terminal (official install.sh) in /content/strata_terminal. Returns the strata binary."""
    def ready(root):
        return _executable(os.path.join(root, "bin", "strata"), os.path.join(root, ".local", "bin", "strata"))

    def build(root):
        log("Installing Strata Terminal (official install.sh)...")
        script = os.path.join(root, "install.sh")
        run(f"curl -fsSL {STRATA_INSTALL_SH} -o {script}")
        env_prefix = (f"STRATA_FEED={STRATA_FEED} HOME={root} XDG_DATA_HOME={root}/share XDG_CONFIG_HOME={root}/config "
                      f"XDG_BIN_HOME={root}/bin PATH={root}/bin:$PATH")
        try:
            run(f"{env_prefix} sh {script}")
        except InstallError as e:
            log("Strata install.sh reported an error; looking for the binary anyway:", str(e)[:300])
    return local_tree("strata_terminal", builds_dir, ready, build, refresh, run, log, "Strata Terminal",
                      cache=_versioned("strata_terminal", STRATA_VERSION))


MAYA_BUILD_PARTS = ("build", ".venv")     # what Maya's setup.sh makes at the first start (engine + its venv)


def maya_build_tar(builds_dir, arch):
    """Maya's compiled engine for one GPU architecture. Maya stamps a build with the archs it was compiled for and
    rebuilds for any other, so the A100 (sm80) and G4 (sm120) builds are kept apart."""
    return os.path.join(builds_dir, _versioned("maya_build", MAYA_VERSION) + f"_sm{arch}.tar")


def maya_build_ready(root):
    build = os.path.join(str(root), "build")
    return os.path.isfile(os.path.join(build, "MAYA-BUILD.json")) and any(
        os.path.isfile(os.path.join(build, n)) for n in ("maya", "maya-ds4", "maya-mm3"))


def restore_maya_build(root, builds_dir, arch, run=sh, log=print):
    """Unpack this GPU's saved Maya build into the checkout, so setup.sh starts without compiling (it took over an
    hour on an A100). True when a build is in place afterwards."""
    tar = maya_build_tar(builds_dir, arch)
    if maya_build_ready(root) or not os.path.isfile(tar):
        return maya_build_ready(root)
    log(f"Restoring Maya's sm_{arch} engine build from Drive:", tar)
    try:
        run(f"tar -xf {tar} -C {root}")
    except Exception as e:                     # a tarball cut off mid-upload: setup.sh compiles again
        log(f"Could not unpack the saved Maya build ({e}); it is compiled again at the first start")
    return maya_build_ready(root)


def save_maya_build(root, builds_dir, arch, run=sh, log=print):
    """After a start that compiled Maya's engine, save build/ and .venv/ for this GPU architecture on Drive.
    Skipped when the saved one has the same build stamp. True when a new tarball was written."""
    if not (arch and builds_dir and maya_build_ready(root)):
        return False
    stamp = Path(root, "build", "MAYA-BUILD.json").read_text()
    tar = maya_build_tar(builds_dir, arch)
    side = tar + ".json"
    if os.path.isfile(tar) and os.path.isfile(side) and Path(side).read_text() == stamp:
        return False
    parts = " ".join(p for p in MAYA_BUILD_PARTS if os.path.exists(os.path.join(str(root), p)))
    os.makedirs(builds_dir, exist_ok=True)
    log(f"Saving Maya's sm_{arch} engine build to Drive (later starts skip the compile):", tar)
    run(f"tar -cf {tar}.tmp -C {root} {parts} && mv {tar}.tmp {tar}")
    with open(side, "w") as f:
        f.write(stamp)
    return True


def ensure_maya(builds_dir, refresh=False, run=sh, log=print, arch=None):
    """Project Maya (git clone; its ./setup.sh runs at the first model load) in /content/project-maya.

    With `arch`, this GPU's saved engine build (see save_maya_build) is unpacked into it as well."""
    def ready(root):
        ok = os.path.isfile(os.path.join(root, "setup.sh")) and os.path.exists(os.path.join(root, ".pybin", "python3"))
        return root if ok else None

    def build(root):
        log("Cloning Project Maya (mw00/project-maya)...")
        branch = f"--branch {MAYA_VERSION} " if MAYA_VERSION else ""
        run(f"rm -rf {root} && git clone -q --depth 1 {branch}{MAYA_REPO} {root} && chmod +x {root}/*.sh")
        # maya.sh makes its venv with `python3 -m venv`; Colab's Python has no working ensurepip, so a uv-managed
        # Python 3.12 (inside root, so the Drive tarball carries it) goes first on Maya's PATH as python3.
        uv = f"UV_PYTHON_INSTALL_DIR={root}/.python UV_CACHE_DIR=/content/.uv-cache {sys.executable} -m uv"
        run(f"{sys.executable} -m pip install -q uv && {uv} python install 3.12")
        py = run(f"{uv} python find --managed-python 3.12").strip().splitlines()[-1]
        run(f"mkdir -p {root}/.pybin && ln -sf {py} {root}/.pybin/python3 && ln -sf {py} {root}/.pybin/python")
    root = local_tree("project-maya", builds_dir, ready, build, refresh, run, log, "Project Maya",
                      cache=_versioned("project-maya", MAYA_VERSION))
    if arch and not refresh:
        restore_maya_build(root, builds_dir, arch, run=run, log=log)
    return root


TENSORFOLD_VERSION = pin("tensorfold")
TENSORFOLD_URL = (
    f"https://github.com/ashhart/TensorFold/releases/download/v{TENSORFOLD_VERSION}/"
    f"tensorfold-{TENSORFOLD_VERSION}-linux-x86_64.tar.gz"
)
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
COMFYUI_PIN = pin("comfyui")
COMFYUI_KEEP = ("torch", "torchvision", "torchaudio")   # Colab's CUDA builds stay; PyPI's would replace them
SPARSA_VERSION = pin("sparsa")
SPARSA_REPO = "https://github.com/LyJonathan/Sparsa.git"
SPARSA_TARBALL = (
    f"https://github.com/LyJonathan/Sparsa/releases/download/{SPARSA_VERSION}/"
    f"sparsa-{SPARSA_VERSION}-linux-x64-cuda.tar.gz"
)
COMFYUI_PKGS = ("torchsde", "einops", "transformers", "safetensors", "aiohttp",
                "pyyaml", "Pillow", "scipy", "tqdm", "psutil", "huggingface_hub")


def ensure_tensorfold(builds_dir, refresh=False, run=sh, log=print):
    """TensorFold's native linux-x86_64 release in /content/tensorfold-<version>. Returns tensorfold-native."""
    name = f"tensorfold-{TENSORFOLD_VERSION}"

    def ready(root):
        for dirpath, _, files in os.walk(root):
            if "tensorfold-native" in files and os.access(os.path.join(dirpath, "tensorfold-native"), os.X_OK):
                return os.path.join(dirpath, "tensorfold-native")
        return None

    def build(root):
        log(f"Downloading TensorFold {TENSORFOLD_VERSION} (official linux-x86_64 release)...")
        run(f"curl -fsSL {TENSORFOLD_URL} -o /tmp/tensorfold.tar.gz && "
            f"tar -xzf /tmp/tensorfold.tar.gz -C {root} --strip-components=1 && rm -f /tmp/tensorfold.tar.gz")
    return local_tree(name, builds_dir, ready, build, refresh, run, log, "TensorFold")


def version_key(tag):
    """Sort key of a release tag like v0.45.4."""
    return tuple(int(x) for x in re.findall(r"\d+", tag))


def comfyui_wheelhouse(builds_dir, pin=None):
    pin = pin or COMFYUI_PIN
    return os.path.join(builds_dir, f"comfyui_{pin}_gpu_py{sys.version_info.major}{sys.version_info.minor}")


def comfyui_requirements(root, dest="/content/comfyui-requirements.txt"):
    """ComfyUI's requirements.txt without torch / torchvision / torchaudio (Colab's own builds are kept)."""
    src = os.path.join(root, "requirements.txt")
    if not os.path.isfile(src):
        return None
    keep = []
    for line in open(src).read().splitlines():
        name = re.split(r"[<>=!~\[; ]", line.strip(), maxsplit=1)[0].lower()
        if name not in COMFYUI_KEEP:
            keep.append(line)
    with open(dest, "w") as f:
        f.write("\n".join(keep) + "\n")
    return dest


def ensure_comfyui(builds_dir, refresh=False, run=sh, log=print, pin=None):
    """Clone ComfyUI + install deps from Drive wheels. Returns ComfyUI root path.

    `pin` is a model's own ComfyUI release (its catalog `comfyui_pin`) when it needs nodes newer than COMFYUI_PIN;
    each release is its own folder and wheel cache, so the tested models keep theirs."""
    pin = pin or COMFYUI_PIN
    pip = f"{sys.executable} -m pip"

    def ready(root):
        return root if os.path.isfile(os.path.join(root, "main.py")) else None

    def build(root):
        log("Cloning ComfyUI...")
        try:
            run(f"rm -rf {root} && git clone -q --depth 1 --branch {pin} {COMFYUI_REPO} {root}")
        except InstallError:
            log(f"Tag {pin} unavailable; cloning default branch...")
            run(f"rm -rf {root} && git clone -q --depth 1 {COMFYUI_REPO} {root}")
    root = local_tree(f"ComfyUI-{pin}", builds_dir, ready, build, refresh, run, log, "ComfyUI")
    wheels = comfyui_wheelhouse(builds_dir, pin)
    pkgs = " ".join(COMFYUI_PKGS)
    # Prefer ComfyUI's requirements.txt when present (less torch); fall back to the pinned package list.
    req = comfyui_requirements(root) or ""
    cached = bool(glob.glob(f"{wheels}/*.whl")) and not refresh
    if cached:
        log("Installing ComfyUI deps from Drive wheels...")
        try:
            if req:
                run(f"{pip} install -q --no-index --find-links {wheels} -r {req}")
            else:
                run(f"{pip} install -q --no-index --find-links {wheels} {pkgs}")
            return root
        except InstallError as e:
            log("Cached ComfyUI install failed, rebuilding:", str(e)[:300])
            cached = False
    log("Downloading ComfyUI Python deps (one time; saved to Drive)...")
    os.makedirs(wheels, exist_ok=True)
    try:
        if req:
            run(f"{pip} wheel -q -w {wheels} -r {req}")
            run(f"{pip} install -q --no-index --find-links {wheels} -r {req}")
        else:
            run(f"{pip} wheel -q -w {wheels} {pkgs}")
            run(f"{pip} install -q --no-index --find-links {wheels} {pkgs}")
    except InstallError:
        log("Wheel cache incomplete; falling back to a direct pip install...")
        if req:
            run(f"{pip} install -q -r {req}")
        else:
            run(f"{pip} install -q {pkgs}")
    log("ComfyUI ready at", root)
    return root


def ensure_sparsa(builds_dir, refresh=False, run=sh, log=print):
    """Sparsa (git clone + the linux CUDA engine release) in /content/Sparsa. Returns its root."""
    def ready(root):
        ok = os.path.isfile(os.path.join(root, "sparsa.py")) and _executable(os.path.join(root, "bin", "strata-minimax"))
        return root if ok else None

    def build(root):
        log("Cloning Sparsa (LyJonathan/Sparsa)...")
        run(f"rm -rf {root} && git clone -q --depth 1 {SPARSA_REPO} {root}")
        log(f"Downloading Sparsa engine {SPARSA_VERSION}...")
        bin_dir = os.path.join(root, "bin")
        run(f"mkdir -p {bin_dir} && curl -fsSL {SPARSA_TARBALL} -o /tmp/sparsa.tar.gz && "
            f"tar -xzf /tmp/sparsa.tar.gz -C {bin_dir} && rm -f /tmp/sparsa.tar.gz")
        engine = os.path.join(bin_dir, "strata-minimax")
        if not os.path.isfile(engine):                    # the archive may nest it one level down
            for dirpath, _, files in os.walk(bin_dir):
                if "strata-minimax" in files:
                    shutil.copy2(os.path.join(dirpath, "strata-minimax"), engine)
                    break
        if os.path.isfile(engine):
            run(f"chmod +x {engine}")
    root = local_tree("Sparsa", builds_dir, ready, build, refresh, run, log, "Sparsa")
    req = os.path.join(root, "requirements.txt")
    if os.path.isfile(req):
        try:
            run(f"{sys.executable} -m pip install -q -r {req}")
        except InstallError as e:
            log("Sparsa requirements install reported an error:", str(e)[:300])
    return root


# ---- HyperQwen (patched vLLM 0.32.0 for Qwen3.8-27B) ----------------------------------------------------
HYPERQWEN_REPO = "https://github.com/syv-ai/HyperQwen.git"
HYPERQWEN_PIN = pin("hyperqwen")
HYPERQWEN_ROOT = "/content/HyperQwen"
# docs/install.md (bare metal): vLLM pin plus the CUDA 13 nvcc that FlashInfer's JIT needs.
HYPERQWEN_PIP = ("vllm==0.32.0 huggingface_hub hf_transfer ninja --extra-index-url https://flashinfer.ai/whl/ "
                 "flashinfer-cubin==0.7.0.post1 pandas nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 "
                 "nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88")


HYPERQWEN_PYTHON = "3.12"     # what HyperQwen documents; uv supplies it, whatever Python the runtime has


def hyperqwen_tarball(builds_dir, ref=HYPERQWEN_PIN):
    return os.path.join(builds_dir, f"hyperqwen_{ref[:10]}_py{HYPERQWEN_PYTHON.replace('.', '')}.tar")


def hyperqwen_build_commands(root=HYPERQWEN_ROOT, ref=HYPERQWEN_PIN, python=None):
    """Clone the pinned HyperQwen, build its venv and apply its vLLM patches (docs/install.md).

    Colab's Python has no working ensurepip and is newer than the 3.12 HyperQwen documents, so uv builds the
    venv with its own Python 3.12, installed inside `root` so the Drive tarball carries it."""
    python = python or sys.executable
    venv_py = f"{root}/venv/bin/python"
    uv = f"UV_PYTHON_INSTALL_DIR={root}/.python UV_CACHE_DIR=/content/.uv-cache {python} -m uv"
    return [
        f"rm -rf {root} && mkdir -p {root}",
        f"cd {root} && git init -q && git fetch -q --depth 1 {HYPERQWEN_REPO} {ref} && git checkout -q FETCH_HEAD",
        f"{python} -m pip install -q uv",
        f"cd {root} && {uv} venv --seed --python {HYPERQWEN_PYTHON} --managed-python venv",
        f"cd {root} && {uv} pip install --python venv/bin/python --index-strategy unsafe-best-match "
        f"{HYPERQWEN_PIP}",
        f"cd {root} && SP=$({venv_py} -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1)"
        f" && bash patches/apply.sh \"$SP\"",
    ]


def ensure_hyperqwen(builds_dir, refresh=False, run=sh, log=print, root=HYPERQWEN_ROOT, ref=HYPERQWEN_PIN,
                     install_now=True):
    """HyperQwen checkout + patched venv at `root`; cached on Drive as one tarball (several GB).

    Needs no GPU, so a CPU runtime can build the tarball (install_now=False returns once it is on Drive)."""
    tar = hyperqwen_tarball(builds_dir, ref)
    if not install_now and os.path.isfile(tar) and not refresh:
        log("HyperQwen build already on Drive:", tar)
        return tar
    if os.path.isfile(os.path.join(root, "venv", "bin", "vllm")) and not refresh:
        return root
    if os.path.isfile(tar) and not refresh:
        log("Restoring HyperQwen (patched vLLM venv) from Drive:", tar)
        run(f"rm -rf {root} && mkdir -p {os.path.dirname(root)} && tar -xf {tar} -C {os.path.dirname(root)}")
        return root
    log(f"Building HyperQwen {ref[:10]} (vLLM 0.32.0 + patches); first run takes 10-20 minutes...")
    for cmd in hyperqwen_build_commands(root, ref):
        run(cmd)
    try:
        run(f"cd {root} && bash xgrammar/install.sh")      # optional upstream tool-call fix
    except InstallError as e:
        log("HyperQwen xgrammar/install.sh failed (optional, tool calls still work):", str(e)[:300])
    os.makedirs(builds_dir, exist_ok=True)
    run(f"tar -cf {tar}.tmp -C {os.path.dirname(root)} {os.path.basename(root)} && mv {tar}.tmp {tar}")
    log("Cached HyperQwen on Drive:", tar)
    return root if install_now else tar


# ---- NInfer (native C++/CUDA, Blackwell sm_120 only) -----------------------------------------------------
NINFER_REPO = "https://github.com/Neroued/ninfer.git"
NINFER_PIN = pin("ninfer")
NINFER_SRC = "/content/ninfer-src"
NINFER_ROOT = "/content/ninfer"            # apps/ + lib/ (restored from Drive)
NINFER_APT = ("ninja-build pkg-config libjpeg-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev "
              "libhwloc-dev libssl-dev zlib1g-dev liburing-dev libcurl4-openssl-dev")
# Ubuntu 24.04 (current Colab) ships liburing 2.5 / libcurl 8.5; older images get them built from source.
NINFER_DEPS = "/content/ninfer-deps"       # liburing / libcurl newer than Ubuntu 22.04 ships
LIBURING_TAG = "liburing-2.9"
CURL_VERSION = "8.11.1"
MIN_DRIVER_FOR_CUDA13 = 580


def ninfer_tarball(builds_dir, ref=NINFER_PIN):
    return os.path.join(builds_dir, f"ninfer_{ref[:10]}_sm120.tar")


def _pkg_version(name, run=sh):
    try:
        return tuple(int(x) for x in run(f"pkg-config --modversion {name}").strip().split(".")[:2])
    except (InstallError, ValueError):
        return None


def _driver_major(run=sh):
    try:
        return int(run("nvidia-smi --query-gpu=driver_version --format=csv,noheader").split(".")[0])
    except (InstallError, ValueError):
        return None


def ninfer_dep_commands(have_uring, have_curl, prefix=NINFER_DEPS):
    cmds = []
    if not have_uring or have_uring < (2, 5):
        cmds += [f"rm -rf /tmp/liburing && git clone -q --depth 1 --branch {LIBURING_TAG} "
                 f"https://github.com/axboe/liburing.git /tmp/liburing",
                 f"cd /tmp/liburing && ./configure --prefix={prefix} && make -j$(nproc) -C src && make install"]
    if not have_curl or have_curl < (7, 85):
        cmds += [f"cd /tmp && curl -fsSL https://curl.se/download/curl-{CURL_VERSION}.tar.xz -o curl.tar.xz"
                 f" && rm -rf curl-{CURL_VERSION} && tar -xf curl.tar.xz",
                 f"cd /tmp/curl-{CURL_VERSION} && ./configure --prefix={prefix} --with-openssl --disable-static"
                 f" --without-libpsl --without-brotli --without-zstd --without-libidn2 --without-nghttp2 -q"
                 f" && make -j$(nproc) && make install"]
    return cmds


def ensure_ninfer(builds_dir, arch, refresh=False, run=sh, log=print, ref=NINFER_PIN, root=NINFER_ROOT,
                  install_now=True):
    """Build NInfer's ninfer-serve for sm_120a and cache apps/ + lib/ on Drive. Returns the binary path.

    `arch` is the GPU it is built for, which need not be present: a CPU runtime can compile for the G4
    (install_now=False returns the Drive tarball and skips the restore)."""
    if str(arch) != "120":
        raise InstallError(f"NInfer builds only for Blackwell sm_120 (G4 runtime); this GPU is sm_{arch}.")
    binary = os.path.join(root, "apps", "ninfer-serve")
    tar = ninfer_tarball(builds_dir, ref)
    if not install_now and os.path.isfile(tar) and not refresh:
        log("NInfer build already on Drive:", tar)
        return tar
    if os.path.isfile(binary) and not refresh:
        return binary
    run(f"apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y {NINFER_APT}")
    if os.path.isfile(tar) and not refresh:
        log("Restoring NInfer from Drive:", tar)
        run(f"rm -rf {root} && mkdir -p {root} && tar -xf {tar} -C {root}")
        return binary
    cuda_home = ensure_cuda13("NInfer", run=run, log=log)
    run(f"{sys.executable} -m pip install -q 'cmake>=3.28'")
    for cmd in ninfer_dep_commands(_pkg_version("liburing", run), _pkg_version("libcurl", run)):
        run(cmd)
    env = (f"PATH={cuda_home}/bin:$PATH CUDACXX={cuda_home}/bin/nvcc "
           f"PKG_CONFIG_PATH={NINFER_DEPS}/lib/pkgconfig:$PKG_CONFIG_PATH "
           f"LD_LIBRARY_PATH={NINFER_DEPS}/lib:$LD_LIBRARY_PATH")
    log(f"Building NInfer {ref[:10]} for sm_120a (first run takes a while)...")
    run(f"rm -rf {NINFER_SRC} && mkdir -p {NINFER_SRC} && cd {NINFER_SRC} && git init -q && "
        f"git fetch -q --depth 1 {NINFER_REPO} {ref} && git checkout -q FETCH_HEAD && "
        f"git submodule update --init --depth 1 -q || true")
    run(f"cd {NINFER_SRC} && {env} cmake --preset rtx5090-release && {env} cmake --build build/rtx5090-release -j")
    out = os.path.join(NINFER_SRC, "build", "rtx5090-release")
    run(f"rm -rf {root} && mkdir -p {root}/lib && cp -a {out}/apps {root}/ && "
        f"(cp -a {out}/lib/. {root}/lib/ 2>/dev/null || true)")
    # Bundle the non-system shared libraries (CUDA 13 runtime, liburing, libcurl) so a restored build runs.
    run(f"{env} ldd {root}/apps/ninfer-serve | awk '/=> \\// {{print $3}}' | "
        f"grep -E '^(/usr/local|{NINFER_DEPS})' | xargs -r -I{{}} cp -L {{}} {root}/lib/")
    os.makedirs(builds_dir, exist_ok=True)
    run(f"tar -cf {tar}.tmp -C {root} apps lib && mv {tar}.tmp {tar}")
    log("Cached NInfer on Drive:", tar)
    return binary if install_now else tar


def ensure_cuda13(what, run=sh, log=print):
    """A CUDA 13 toolkit (nvcc) for engines compiled for Blackwell. Returns its CUDA_HOME.

    CUDA 13.2.0 / 13.2.1 miscompile some sm_120 kernels (Strata #892 / #968, fixed in 13.2.2), so those are
    upgraded too."""
    driver = _driver_major(run)
    if driver is not None and driver < MIN_DRIVER_FOR_CUDA13:
        raise InstallError(f"{what} needs CUDA 13, which needs NVIDIA driver {MIN_DRIVER_FOR_CUDA13}+; "
                           f"this runtime has {driver}.")
    out = _try(run, "nvcc --version") or _try(run, "/usr/local/cuda/bin/nvcc --version")
    nvcc = gpu.nvcc_version(out)
    build = re.search(r"V13\.2\.(\d+)", out or "")
    bad = nvcc == (13, 2) and bool(build) and int(build.group(1)) < 86      # 13.2.86 is CUDA 13.2.2
    if nvcc and nvcc >= (13, 0) and not bad:
        return "/usr/local/cuda"
    log(f"Installing the CUDA 13 toolkit (runtime has nvcc {nvcc})...")
    run("cd /tmp && . /etc/os-release && distro=ubuntu$(echo $VERSION_ID | tr -d .) && "
        "curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb"
        " -o cuda-keyring.deb && dpkg -i cuda-keyring.deb && apt-get -qq update && "
        "DEBIAN_FRONTEND=noninteractive apt-get -qq install -y cuda-toolkit-13")
    homes = sorted(glob.glob("/usr/local/cuda-13*"), key=lambda p: [int(x) for x in p.rsplit("-", 1)[1].split(".")
                                                                     if x.isdigit()])
    return homes[-1] if homes else "/usr/local/cuda"


def _uv_venv_commands(root, venv, python_version="3.12", python=None):
    """A venv with pip at root/venv made by uv (Colab's Python has no working ensurepip), its Python inside root."""
    python = python or sys.executable
    uv = f"UV_PYTHON_INSTALL_DIR={root}/.python UV_CACHE_DIR=/content/.uv-cache {python} -m uv"
    return [f"{python} -m pip install -q uv",
            f"cd {root} && {uv} venv --seed --python {python_version} --managed-python {venv}"]


# ---- Basalt (Strata fork for Blackwell: Qwen3.8-Flash-Next NVFP4 .basalt packs) ------------------------------
BASALT_REPO = "https://github.com/jesdga95/basalt.git"
BASALT_PIN = pin("basalt")
BASALT_ROOT = "/content/basalt"


def basalt_tarball(builds_dir, ref=BASALT_PIN):
    return os.path.join(builds_dir, f"basalt_{ref[:10]}_sm120.tar")


def ensure_basalt(builds_dir, arch, refresh=False, run=sh, log=print, ref=BASALT_PIN, root=BASALT_ROOT,
                  install_now=True):
    """Build Basalt (bin/basalt + its Python server venv) for sm_120 and cache it on Drive. Returns root.

    Its README: cmake -S . -B build; cmake --build build -j; pip install -r requirements.txt. Needs CUDA 13
    and GCC 13+ (Ubuntu 24.04 has both after the toolkit install)."""
    if str(arch) != "120":
        raise InstallError(f"Basalt builds only for Blackwell sm_120 (G4 runtime); this GPU is sm_{arch}.")
    tar = basalt_tarball(builds_dir, ref)
    if not install_now and os.path.isfile(tar) and not refresh:
        log("Basalt build already on Drive:", tar)
        return tar
    if os.path.isfile(os.path.join(root, "bin", "basalt")) and not refresh:
        return root
    if os.path.isfile(tar) and not refresh:
        log("Restoring Basalt from Drive:", tar)
        run(f"rm -rf {root} && mkdir -p {os.path.dirname(root)} && tar -xf {tar} -C {os.path.dirname(root)}")
        return root
    cuda_home = ensure_cuda13("Basalt", run=run, log=log)
    run(f"{sys.executable} -m pip install -q 'cmake>=3.28' ninja")
    log(f"Building Basalt {ref[:10]} for sm_120 (first run takes a while)...")
    run(f"rm -rf {root} && mkdir -p {root} && cd {root} && git init -q && "
        f"git fetch -q --depth 1 {BASALT_REPO} {ref} && git checkout -q FETCH_HEAD")
    env = f"PATH={cuda_home}/bin:$PATH CUDACXX={cuda_home}/bin/nvcc CUDA_HOME={cuda_home}"
    run(f"cd {root} && {env} cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && "
        f"{env} cmake --build build -j")
    for cmd in _uv_venv_commands(root, "venv"):
        run(cmd)
    run(f"cd {root} && venv/bin/python -m pip install -q -r requirements.txt")
    # Bundle the CUDA 13 runtime libraries the binaries link, so a restored build runs without the toolkit.
    run(f"mkdir -p {root}/lib && for b in {root}/bin/*; do ldd $b 2>/dev/null; done | awk '/=> \\// {{print $3}}' "
        f"| grep -E '^/usr/local' | sort -u | xargs -r -I{{}} cp -L {{}} {root}/lib/")
    os.makedirs(builds_dir, exist_ok=True)
    run(f"tar -cf {tar}.tmp -C {os.path.dirname(root)} --exclude={os.path.basename(root)}/build "
        f"{os.path.basename(root)} && mv {tar}.tmp {tar}")
    log("Cached Basalt on Drive:", tar)
    return root if install_now else tar


# ---- Strata NVFP4 fork (sergqwer/strata-nvfp4: Qwen3.8-Flash-Next NVFP4 GPTQ) --------------------------------
STRATA_NVFP4_REPO = "https://github.com/sergqwer/strata-nvfp4.git"
STRATA_NVFP4_PIN = pin("strata_nvfp4")
STRATA_NVFP4_ROOT = "/content/strata-nvfp4"


def strata_nvfp4_tarball(builds_dir, ref=STRATA_NVFP4_PIN):
    return os.path.join(builds_dir, f"strata_nvfp4_{ref[:10]}_sm120.tar")


def ensure_strata_nvfp4(builds_dir, arch, refresh=False, run=sh, log=print, ref=STRATA_NVFP4_PIN,
                        root=STRATA_NVFP4_ROOT, install_now=True):
    """The fork's checkout with a uv venv and the CUDA 13 toolkit. Returns root.

    Its ./setup.sh compiles the engine on Linux and checks the model files, so the first build happens at
    the first start on the GPU runtime (strata_nvfp4_launch), which then caches the whole folder on Drive."""
    if str(arch) != "120":
        raise InstallError("This Strata NVFP4 setup is for the G4 runtime (Blackwell sm_120); "
                           f"this GPU is sm_{arch}.")
    tar = strata_nvfp4_tarball(builds_dir, ref)
    if not install_now:
        log("Strata NVFP4 is built on the GPU runtime at its first start" +
            (f" (cached: {tar})" if os.path.isfile(tar) else ""))
        return tar
    if os.path.isfile(os.path.join(root, ".venv", "bin", "python")) and not refresh:
        return root
    if os.path.isfile(tar) and not refresh:
        log("Restoring Strata NVFP4 from Drive:", tar)
        run(f"rm -rf {root} && mkdir -p {os.path.dirname(root)} && tar -xf {tar} -C {os.path.dirname(root)}")
        return root
    ensure_cuda13("Strata NVFP4", run=run, log=log)
    run(f"rm -rf {root} && mkdir -p {root} && cd {root} && git init -q && "
        f"git fetch -q --depth 1 {STRATA_NVFP4_REPO} {ref} && git checkout -q FETCH_HEAD")
    for cmd in _uv_venv_commands(root, ".venv"):
        run(cmd)
    log("Strata NVFP4 checked out; its engine is compiled at the first start (cached on Drive after that)")
    return root


def _try(run, cmd):
    try:
        return run(cmd)
    except InstallError:
        return ""
