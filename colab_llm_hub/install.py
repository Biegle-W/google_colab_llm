"""Engine installs, cached on Google Drive so a new Colab session restores them instead of rebuilding.

llama.cpp is compiled for the detected GPU architecture and saved as a tarball of build/bin. vLLM is a pip
package: its wheels are saved in a wheelhouse on Drive and installed offline in later sessions.
"""
import glob
import os
import subprocess
import sys

from . import gpu

LLAMA_URL = "https://github.com/ggml-org/llama.cpp"
# The commit the reference setup validated its llama.cpp flags against (fetched by SHA, so it stays reproducible).
LLAMA_PIN = "4c3c5810f87efe177949f1b85ae509d38a2a311f"
SRC = "/content/llama.cpp"


class InstallError(RuntimeError):
    pass


def sh(cmd, log_path="/content/install.log"):
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
