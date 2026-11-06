"""Engine installs, cached on Google Drive so a new Colab session restores them instead of rebuilding.

llama.cpp is compiled for the detected GPU architecture and saved as a tarball of build/bin. vLLM / SGLang /
Diffusers wheels are saved in a wheelhouse on Drive and installed offline in later sessions. Install logs
also go under the Drive workspace ``logs/`` when ``COLAB_LLM_DRIVE`` is set.
"""
import glob
import os
import shutil
import subprocess
import sys

from . import gpu

LLAMA_URL = "https://github.com/ggml-org/llama.cpp"
# The commit the reference setup validated its llama.cpp flags against (fetched by SHA, so it stays reproducible).
LLAMA_PIN = "4c3c5810f87efe177949f1b85ae509d38a2a311f"
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


def ensure_llama(builds_dir, arch, ref=LLAMA_PIN, rebuild=False, run=sh, log=print, src=SRC):
    """Return the path to llama-server for this GPU, restoring a cached build or building (and caching) one."""
    os.environ["PATH"] += ":/usr/local/cuda/bin"
    cuda = gpu.nvcc_version(subprocess.run("nvcc --version", shell=True, capture_output=True, text=True).stdout)
    if cuda is None:
        raise InstallError("nvcc (the CUDA compiler) was not found on this runtime.")
    arch_cmake = gpu.cmake_arch(arch, cuda)
    if arch_cmake != str(arch):
        log(f"CUDA {cuda[0]}.{cuda[1]} cannot compile sm_{arch}; using {arch_cmake} (works, may be slightly slower).")
    os.makedirs(builds_dir, exist_ok=True)
    tarball, binary = llama_tarball(builds_dir, ref, arch_cmake), f"{src}/build/bin/llama-server"

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
    if not works():
        raise InstallError("llama-server was built but does not start; see /content/install.log")
    return binary


def vllm_wheelhouse(builds_dir):
    return os.path.join(builds_dir, f"vllm_gpu_py{sys.version_info.major}{sys.version_info.minor}")


def ensure_vllm(builds_dir, refresh=False, run=sh, log=print):
    """Install vLLM from wheels cached on Drive (downloading them once)."""
    wheels = vllm_wheelhouse(builds_dir)
    pip = f"{sys.executable} -m pip"
    cached = bool(glob.glob(f"{wheels}/*.whl")) and not refresh
    if cached:
        log("Installing vLLM from the wheels saved on Drive...")
        try:
            run(f"{pip} install -q --no-index --find-links {wheels} vllm")
        except InstallError as e:
            log("Cached install failed, rebuilding the wheel cache:", str(e)[:300])
            cached = False
    if not cached:
        log("Downloading vLLM and its dependencies (several GB, one time)...")
        os.makedirs(wheels, exist_ok=True)
        run(f"{pip} wheel -q -w {wheels} vllm")
        run(f"{pip} install -q --no-index --find-links {wheels} vllm")
    return run(f"{pip} show vllm").split("Version:")[-1].split()[0]


# Diffusers stack used for text-to-image. Torch is already present on Colab GPU images; we cache the rest.
DIFFUSERS_PKGS = ("diffusers", "transformers", "accelerate", "safetensors", "sentencepiece",
                  "protobuf", "Pillow", "bitsandbytes", "huggingface_hub", "hf_xet")


def diffusers_wheelhouse(builds_dir):
    return os.path.join(builds_dir, f"diffusers_gpu_py{sys.version_info.major}{sys.version_info.minor}")


def ensure_diffusers(builds_dir, refresh=False, run=sh, log=print):
    """Install Diffusers (+ transformers/accelerate/bitsandbytes) from wheels cached on Drive."""
    wheels = diffusers_wheelhouse(builds_dir)
    pip = f"{sys.executable} -m pip"
    pkgs = " ".join(DIFFUSERS_PKGS)
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


def cache_hub_package(src_dir, builds_dir, ref="default", log=print):
    """Copy ``colab_llm_hub`` from a fresh git checkout onto Drive for offline restore."""
    src = os.path.join(src_dir, "colab_llm_hub")
    if not os.path.isdir(src):
        raise InstallError(f"no colab_llm_hub package under {src_dir}")
    dest = os.path.join(builds_dir, "hub_pkg", (ref or "default").replace("/", "_")[:80], "colab_llm_hub")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    marker = os.path.join(os.path.dirname(dest), "COMPLETE")
    if os.path.exists(marker):
        os.remove(marker)          # an interrupted refresh must not look complete
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    open(marker, "w").write(ref or "default")
    log("Cached hub package on Drive:", dest)
    return dest


def restore_hub_package(builds_dir, dest_parent, ref="default", log=print):
    """Restore a Drive-cached hub package into dest_parent/colab_llm_hub. Returns True on hit."""
    src = os.path.join(builds_dir, "hub_pkg", (ref or "default").replace("/", "_")[:80], "colab_llm_hub")
    marker = os.path.join(os.path.dirname(src), "COMPLETE")
    if not (os.path.isdir(src) and os.path.isfile(marker)):
        return False
    dest = os.path.join(dest_parent, "colab_llm_hub")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest_parent, exist_ok=True)
    shutil.copytree(src, dest)
    log("Restored hub package from Drive:", src)
    return True


def sglang_wheelhouse(builds_dir):
    return os.path.join(builds_dir, f"sglang_gpu_py{sys.version_info.major}{sys.version_info.minor}")


def ensure_sglang(builds_dir, refresh=False, run=sh, log=print):
    """Install SGLang from wheels cached on Drive (chat serving; diffusion/H3 is separate)."""
    wheels = sglang_wheelhouse(builds_dir)
    pip = f"{sys.executable} -m pip"
    cached = bool(glob.glob(f"{wheels}/*.whl")) and not refresh
    if cached:
        log("Installing SGLang from the wheels saved on Drive...")
        try:
            run(f"{pip} install -q --no-index --find-links {wheels} 'sglang[all]'")
        except InstallError as e:
            log("Cached SGLang install failed, rebuilding the wheel cache:", str(e)[:300])
            cached = False
    if not cached:
        log("Downloading SGLang and its dependencies (several GB, one time)...")
        os.makedirs(wheels, exist_ok=True)
        try:
            run(f"{pip} wheel -q -w {wheels} 'sglang[all]'")
            run(f"{pip} install -q --no-index --find-links {wheels} 'sglang[all]'")
        except InstallError:
            log("Wheel cache incomplete; falling back to a direct pip install of sglang...")
            run(f"{pip} install -q -U 'sglang[all]'")
    return run(f"{pip} show sglang").split("Version:")[-1].split()[0]


STRATA_INSTALL_SH = "https://github.com/Niko1221/Strata/releases/latest/download/install.sh"
MAYA_REPO = "https://github.com/mw00/project-maya.git"


def ensure_strata(builds_dir, refresh=False, run=sh, log=print):
    """Install the Strata Terminal into Drive builds/ (official install.sh). Returns strata binary path."""
    root = os.path.join(builds_dir, "strata_terminal")
    marker = os.path.join(root, "COMPLETE")
    os.makedirs(root, exist_ok=True)
    # Prefer an already-cached binary under the Drive install tree.
    cached = None
    for cand in (os.path.join(root, "bin", "strata"),
                 os.path.join(root, ".local", "bin", "strata"),
                 os.path.join(root, "strata")):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            cached = cand
            break
    if cached and not refresh and os.path.isfile(marker):
        log("Restoring Strata Terminal from Drive:", cached)
        return cached
    log("Installing Strata Terminal (official install.sh) into Drive builds/...")
    script = os.path.join(root, "install.sh")
    run(f"curl -fsSL {STRATA_INSTALL_SH} -o {script}")
    # Isolate install under our Drive tree so later Colab sessions can restore it.
    env_prefix = (
        f"HOME={root} XDG_DATA_HOME={root}/share XDG_CONFIG_HOME={root}/config "
        f"XDG_BIN_HOME={root}/bin PATH={root}/bin:$PATH"
    )
    try:
        run(f"{env_prefix} sh {script}")
    except InstallError as e:
        log("Strata install.sh reported an error; searching for a binary anyway:", str(e)[:300])
    for cand in (os.path.join(root, "bin", "strata"),
                 os.path.join(root, ".local", "bin", "strata"),
                 shutil.which("strata") or ""):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            open(marker, "w").write(cand)
            log("Strata binary:", cand)
            return cand
    raise InstallError(
        "Strata Terminal install finished without a usable `strata` binary. "
        "See logs/install.log and https://github.com/Niko1221/Strata/blob/main/docs/INSTALL.md"
    )


def ensure_maya(builds_dir, refresh=False, run=sh, log=print):
    """Clone Project Maya into Drive builds/ (official git + ./setup.sh on first model load)."""
    root = os.path.join(builds_dir, "project-maya")
    marker = os.path.join(root, "COMPLETE")
    setup = os.path.join(root, "setup.sh")
    if os.path.isfile(setup) and os.path.isfile(marker) and not refresh:
        log("Restoring Project Maya from Drive:", root)
        return root
    log("Cloning Project Maya (mw00/project-maya) onto Drive...")
    parent = os.path.dirname(root)
    os.makedirs(parent, exist_ok=True)
    if os.path.isdir(root):
        shutil.rmtree(root)
    run(f"git clone --depth 1 {MAYA_REPO} {root}")
    if not os.path.isfile(setup):
        raise InstallError(f"Project Maya clone missing setup.sh under {root}")
    run(f"chmod +x {setup}")
    open(marker, "w").write(MAYA_REPO)
    log("Project Maya ready at", root, "(first model load runs ./setup.sh)")
    return root


TENSORFOLD_VERSION = "1.2.2"
TENSORFOLD_URL = (
    f"https://github.com/ashhart/TensorFold/releases/download/v{TENSORFOLD_VERSION}/"
    f"tensorfold-{TENSORFOLD_VERSION}-linux-x86_64.tar.gz"
)
COMFYUI_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
COMFYUI_PIN = "v0.3.60"  # recent stable tag; refresh via REFRESH_WHEELS rebuild
SPARSA_VERSION = "v0.2.2-experimental"
SPARSA_REPO = "https://github.com/LyJonathan/Sparsa.git"
SPARSA_TARBALL = (
    f"https://github.com/LyJonathan/Sparsa/releases/download/{SPARSA_VERSION}/"
    f"sparsa-{SPARSA_VERSION}-linux-x64-cuda.tar.gz"
)
COMFYUI_PKGS = ("torchsde", "einops", "transformers", "safetensors", "aiohttp",
                "pyyaml", "Pillow", "scipy", "tqdm", "psutil", "huggingface_hub")


def ensure_tensorfold(builds_dir, refresh=False, run=sh, log=print):
    """Download TensorFold native linux-x86_64 release into Drive builds/. Returns tensorfold-native path."""
    root = os.path.join(builds_dir, f"tensorfold-{TENSORFOLD_VERSION}")
    marker = os.path.join(root, "COMPLETE")
    binary = os.path.join(root, "bin", "tensorfold-native")
    if os.path.isfile(binary) and os.access(binary, os.X_OK) and os.path.isfile(marker) and not refresh:
        log("Restoring TensorFold from Drive:", binary)
        return binary
    log(f"Downloading TensorFold {TENSORFOLD_VERSION} (official linux-x86_64 release)...")
    os.makedirs(root, exist_ok=True)
    tarball = os.path.join(root, f"tensorfold-{TENSORFOLD_VERSION}-linux-x86_64.tar.gz")
    run(f"curl -fsSL {TENSORFOLD_URL} -o {tarball}")
    run(f"tar -xzf {tarball} -C {root} --strip-components=1")
    if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        # Some archives nest bin/ under a versioned directory.
        found = None
        for dirpath, _, files in os.walk(root):
            if "tensorfold-native" in files:
                cand = os.path.join(dirpath, "tensorfold-native")
                if os.access(cand, os.X_OK):
                    found = cand
                    break
        if not found:
            raise InstallError(f"TensorFold archive missing tensorfold-native under {root}")
        binary = found
    open(marker, "w").write(TENSORFOLD_URL)
    log("TensorFold binary:", binary)
    return binary


def comfyui_wheelhouse(builds_dir):
    return os.path.join(builds_dir, f"comfyui_gpu_py{sys.version_info.major}{sys.version_info.minor}")


def ensure_comfyui(builds_dir, refresh=False, run=sh, log=print):
    """Clone ComfyUI + install deps from Drive wheels. Returns ComfyUI root path."""
    root = os.path.join(builds_dir, "ComfyUI")
    marker = os.path.join(root, "COMPLETE")
    main_py = os.path.join(root, "main.py")
    pip = f"{sys.executable} -m pip"
    if os.path.isfile(main_py) and os.path.isfile(marker) and not refresh:
        log("Restoring ComfyUI from Drive:", root)
    else:
        log("Cloning ComfyUI onto Drive...")
        parent = os.path.dirname(root)
        os.makedirs(parent, exist_ok=True)
        if os.path.isdir(root):
            shutil.rmtree(root)
        try:
            run(f"git clone --depth 1 --branch {COMFYUI_PIN} {COMFYUI_REPO} {root}")
        except InstallError:
            log(f"Tag {COMFYUI_PIN} unavailable; cloning default branch...")
            run(f"git clone --depth 1 {COMFYUI_REPO} {root}")
        open(marker, "w").write(COMFYUI_REPO)
    wheels = comfyui_wheelhouse(builds_dir)
    pkgs = " ".join(COMFYUI_PKGS)
    # Prefer ComfyUI's requirements.txt when present; fall back to the pinned package list.
    req = os.path.join(root, "requirements.txt")
    cached = bool(glob.glob(f"{wheels}/*.whl")) and not refresh
    if cached:
        log("Installing ComfyUI deps from Drive wheels...")
        try:
            if os.path.isfile(req):
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
        if os.path.isfile(req):
            run(f"{pip} wheel -q -w {wheels} -r {req}")
            run(f"{pip} install -q --no-index --find-links {wheels} -r {req}")
        else:
            run(f"{pip} wheel -q -w {wheels} {pkgs}")
            run(f"{pip} install -q --no-index --find-links {wheels} {pkgs}")
    except InstallError:
        log("Wheel cache incomplete; falling back to a direct pip install...")
        if os.path.isfile(req):
            run(f"{pip} install -q -r {req}")
        else:
            run(f"{pip} install -q {pkgs}")
    log("ComfyUI ready at", root)
    return root


def ensure_sparsa(builds_dir, refresh=False, run=sh, log=print):
    """Clone Sparsa + unpack the linux CUDA engine release into Drive builds/. Returns Sparsa root."""
    root = os.path.join(builds_dir, "Sparsa")
    marker = os.path.join(root, "COMPLETE")
    script = os.path.join(root, "sparsa.py")
    engine = os.path.join(root, "bin", "strata-minimax")
    if os.path.isfile(script) and os.path.isfile(engine) and os.path.isfile(marker) and not refresh:
        log("Restoring Sparsa from Drive:", root)
        return root
    log("Cloning Sparsa (LyJonathan/Sparsa) onto Drive...")
    parent = os.path.dirname(root)
    os.makedirs(parent, exist_ok=True)
    if os.path.isdir(root):
        shutil.rmtree(root)
    run(f"git clone --depth 1 {SPARSA_REPO} {root}")
    if not os.path.isfile(script):
        raise InstallError(f"Sparsa clone missing sparsa.py under {root}")
    pip = f"{sys.executable} -m pip"
    req = os.path.join(root, "requirements.txt")
    if os.path.isfile(req):
        try:
            run(f"{pip} install -q -r {req}")
        except InstallError as e:
            log("Sparsa requirements install reported an error:", str(e)[:300])
    log(f"Downloading Sparsa engine {SPARSA_VERSION}...")
    tarball = os.path.join(root, f"sparsa-{SPARSA_VERSION}-linux-x64-cuda.tar.gz")
    run(f"curl -fsSL {SPARSA_TARBALL} -o {tarball}")
    bin_dir = os.path.join(root, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    run(f"tar -xzf {tarball} -C {bin_dir}")
    # Flatten if the archive nested strata-minimax one level down.
    if not os.path.isfile(engine):
        for dirpath, _, files in os.walk(bin_dir):
            if "strata-minimax" in files:
                src = os.path.join(dirpath, "strata-minimax")
                if src != engine:
                    shutil.copy2(src, engine)
                break
    if os.path.isfile(engine):
        run(f"chmod +x {engine}")
    else:
        raise InstallError(f"Sparsa release missing strata-minimax under {bin_dir}")
    open(marker, "w").write(SPARSA_TARBALL)
    log("Sparsa ready at", root)
    return root


# ---- HyperQwen (patched vLLM 0.32.0 for Qwen3.8-27B) ----------------------------------------------------
HYPERQWEN_REPO = "https://github.com/syv-ai/HyperQwen.git"
HYPERQWEN_PIN = "6ab9bd05cb84b538a6595d634f3e337ac8a80856"
HYPERQWEN_ROOT = "/content/HyperQwen"
# docs/install.md (bare metal): vLLM pin plus the CUDA 13 nvcc that FlashInfer's JIT needs.
HYPERQWEN_PIP = ("vllm==0.32.0 huggingface_hub hf_transfer ninja --extra-index-url https://flashinfer.ai/whl/ "
                 "flashinfer-cubin==0.7.0.post1 pandas nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 "
                 "nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88")


def hyperqwen_tarball(builds_dir, ref=HYPERQWEN_PIN):
    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    return os.path.join(builds_dir, f"hyperqwen_{ref[:10]}_{py}.tar")


def hyperqwen_build_commands(root=HYPERQWEN_ROOT, ref=HYPERQWEN_PIN, python=None):
    """Clone the pinned HyperQwen, build its venv and apply its vLLM patches (docs/install.md)."""
    python = python or sys.executable
    venv_py = f"{root}/venv/bin/python"
    return [
        f"rm -rf {root} && mkdir -p {root}",
        f"cd {root} && git init -q && git fetch -q --depth 1 {HYPERQWEN_REPO} {ref} && git checkout -q FETCH_HEAD",
        f"cd {root} && {python} -m venv venv",
        f"cd {root} && venv/bin/pip install -q --upgrade pip",
        f"cd {root} && venv/bin/pip install -q {HYPERQWEN_PIP}",
        f"cd {root} && SP=$({venv_py} -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1)"
        f" && bash patches/apply.sh \"$SP\"",
    ]


def ensure_hyperqwen(builds_dir, refresh=False, run=sh, log=print, root=HYPERQWEN_ROOT, ref=HYPERQWEN_PIN):
    """HyperQwen checkout + patched venv at `root`; cached on Drive as one tarball (several GB)."""
    tar = hyperqwen_tarball(builds_dir, ref)
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
    return root


# ---- NInfer (native C++/CUDA, Blackwell sm_120 only) -----------------------------------------------------
NINFER_REPO = "https://github.com/Neroued/ninfer.git"
NINFER_PIN = "a754fcf3bf524494aa927c2f54e0cbfb58f569ab"
NINFER_SRC = "/content/ninfer-src"
NINFER_ROOT = "/content/ninfer"            # apps/ + lib/ (restored from Drive)
NINFER_APT = ("ninja-build pkg-config libjpeg-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev "
              "libhwloc-dev libssl-dev zlib1g-dev")
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


def ensure_ninfer(builds_dir, arch, refresh=False, run=sh, log=print, ref=NINFER_PIN, root=NINFER_ROOT):
    """Build NInfer's ninfer-serve for sm_120a and cache apps/ + lib/ on Drive. Returns the binary path."""
    if str(arch) != "120":
        raise InstallError(f"NInfer builds only for Blackwell sm_120 (G4 runtime); this GPU is sm_{arch}.")
    binary = os.path.join(root, "apps", "ninfer-serve")
    tar = ninfer_tarball(builds_dir, ref)
    if os.path.isfile(binary) and not refresh:
        return binary
    run(f"apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y {NINFER_APT}")
    if os.path.isfile(tar) and not refresh:
        log("Restoring NInfer from Drive:", tar)
        run(f"rm -rf {root} && mkdir -p {root} && tar -xf {tar} -C {root}")
        return binary
    driver = _driver_major(run)
    if driver is not None and driver < MIN_DRIVER_FOR_CUDA13:
        raise InstallError(f"NInfer needs CUDA 13, which needs NVIDIA driver {MIN_DRIVER_FOR_CUDA13}+; "
                           f"this runtime has {driver}.")
    nvcc = gpu.nvcc_version(_try(run, "nvcc --version") or _try(run, "/usr/local/cuda/bin/nvcc --version"))
    cuda_home = "/usr/local/cuda"
    if not nvcc or nvcc < (13, 0):
        log(f"Installing the CUDA 13 toolkit (runtime has nvcc {nvcc})...")
        run("cd /tmp && . /etc/os-release && distro=ubuntu$(echo $VERSION_ID | tr -d .) && "
            "curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb"
            " -o cuda-keyring.deb && dpkg -i cuda-keyring.deb && apt-get -qq update && "
            "DEBIAN_FRONTEND=noninteractive apt-get -qq install -y cuda-toolkit-13")
        cuda_home = sorted(glob.glob("/usr/local/cuda-13*"))[-1] if glob.glob("/usr/local/cuda-13*") else cuda_home
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
    return binary


def _try(run, cmd):
    try:
        return run(cmd)
    except InstallError:
        return ""
