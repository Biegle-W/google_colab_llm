"""Optional: run the dsh harness (DeepSeek Harness) inside the notebook, with its web UI.

The same harness the reference setup runs (@deepseek-ai/dsh 0.2.1-rc.2 on Node.js 24.21.0), pointed at this
runtime's hub over 127.0.0.1, so it needs no tunnel to reach the models. Its web UI is opened through Colab's
own port proxy (only you, signed in to Colab, can open it), and over Tailscale too when that tunnel is up.

dsh's web UI admits a browser only with the token in the link it prints at every start, and only for host names
it was told to trust (--trusted-host), so the link this module prints is the way in.

- Node.js and dsh are installed in /content/dsh-<version> and cached on Drive (engines/dsh-<version>.tar).
- DSH_HOME (sessions, settings) is on Drive (dsh/), so conversations survive the runtime.
- The agent's working folder is on Drive (workspace/), so its files survive too.
- The hub's models reach dsh through a --patch overlay; your own dsh profile settings are left alone.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import install, provider

NODE_VERSION = "24.21.0"
DSH_VERSION = "0.2.1-rc.2"
NODE_URL = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.xz"
PACKAGE_JSON = """{
  "name": "colab-dsh",
  "private": true,
  "version": "1.0.0",
  "dependencies": {"@deepseek-ai/dsh": "%s"},
  "allowScripts": {
    "@deepseek-ai/dsh-subprocess-local@%s": true,
    "node-pty@1.2.0-beta.15": true,
    "koffi@3.1.1": true,
    "protobufjs@7.6.6": true
  }
}
""" % (DSH_VERSION, DSH_VERSION)
PORT = 3080
TS_HTTPS_PORT = 8443
PROVIDER_ID = "colab"
TOKEN_URL_RE = re.compile(r"dsh web: (\S+\?token=([A-Za-z0-9_-]+))")


def name():
    return f"dsh-{DSH_VERSION}"


def paths(root):
    root = Path(root)
    return root / "node" / "bin" / "node", root / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js"


def ensure(builds_dir, refresh=False, run=install.sh, log=print):
    """Node.js + dsh in /content/dsh-<version> (restored from Drive after the first install). Returns root."""
    def ready(root):
        node, bin_js = paths(root)
        return root if node.is_file() and bin_js.is_file() else None

    def build(root):
        log(f"Installing Node.js {NODE_VERSION} and dsh {DSH_VERSION} (one time; cached on Drive)...")
        run(f"mkdir -p {root}/node && curl -fsSL {NODE_URL} -o /tmp/node.tar.xz && "
            f"tar -xJf /tmp/node.tar.xz -C {root}/node --strip-components=1 && rm -f /tmp/node.tar.xz")
        Path(root, "package.json").write_text(PACKAGE_JSON)
        run(f"cd {root} && PATH={root}/node/bin:$PATH npm install --no-audit --no-fund --loglevel=error")
    return install.local_tree(name(), builds_dir, ready, build, refresh, run, log, "dsh")


BWRAP_SHIM = """#!/bin/sh
# dsh confines the agent's commands with bubblewrap: read-only /, writable workspace and private /tmp,
# its own PID namespace. Colab's container refuses to mount a fresh /proc inside that namespace
# ("Can't mount proc on /newroot/proc: Permission denied"), so this passes the existing /proc through
# instead; every other part of dsh's profile is unchanged.
prev=""; done_opts=""
for a in "$@"; do
  shift
  if [ -n "$done_opts" ]; then set -- "$@" "$a"; continue; fi          # the command itself: untouched
  if [ "$prev" = "--proc" ]; then set -- "$@" "/proc" "$a"; prev=""; continue; fi
  if [ "$a" = "--proc" ]; then set -- "$@" "--bind"; prev="--proc"; continue; fi
  [ "$a" = "--" ] && done_opts=1
  set -- "$@" "$a"; prev="$a"
done
exec /usr/bin/bwrap "$@"
"""


def ensure_sandbox(root, run=install.sh, log=print):
    """bubblewrap (apt, every session: system packages are not kept) and the /proc shim. Returns its dir."""
    if not os.path.isfile("/usr/bin/bwrap"):
        log("Installing bubblewrap (dsh's command sandbox)...")
        run("DEBIAN_FRONTEND=noninteractive apt-get -qq install -y bubblewrap || "
            "(apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y bubblewrap)")
    shim_dir = Path(root) / "sandbox-bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "bwrap"
    if not shim.is_file() or shim.read_text() != BWRAP_SHIM:
        shim.write_text(BWRAP_SHIM)
    shim.chmod(0o755)
    return str(shim_dir)


def write_patch(path, models, hub_port, default_model=None):
    """A dsh patch overlay: the hub as provider `colab` (reached on 127.0.0.1) and the default model."""
    import json
    chat = provider._chat_models(models)
    if not chat:
        raise ValueError("the hub serves no chat model for dsh")
    block = provider.provider_dict(f"http://127.0.0.1:{hub_port}", chat)
    block["displayName"] = "Colab (this runtime)"
    default = default_model if default_model in {m["id"] for m in chat} else chat[0]["id"]
    patch = [
        {"id": "llm-pi-ai", "name": "@deepseek-ai/dsh-llm-pi-ai", "config": {"providers": {PROVIDER_ID: block}}},
        {"id": "agent-default-model", "name": "@deepseek-ai/dsh-agent-default-model",
         "config": {"provider": PROVIDER_ID, "model": default}},
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(patch, indent=1))      # JSON is valid YAML
    return default


def web_args(root, patch, port=PORT, public_url=None, trusted_hosts=()):
    node, bin_js = paths(root)
    args = [str(node), str(bin_js), "--profile", "web", "--patch", str(patch), "--no-open",
            "--host", "127.0.0.1", "--port", str(port)]
    if public_url:
        args += ["--public-url", public_url]
    for h in trusted_hosts:
        args += ["--trusted-host", h]
    return args


def parse_token(text):
    m = TOKEN_URL_RE.search(text or "")
    return m.group(2) if m else None


def colab_proxy_url(port=PORT):
    """https URL of Colab's authenticated proxy to a port on this runtime (None outside Colab)."""
    try:
        from google.colab.output import eval_js
        return eval_js(f"google.colab.kernel.proxyPort({int(port)})").rstrip("/") + "/"
    except Exception:
        return None


def host_of(url):
    return re.sub(r"^https?://", "", url).split("/")[0]


def stop(port=PORT):
    subprocess.run(["pkill", "-TERM", "-f", f"[d]sh/lib/bin.js --profile web .*--port {int(port)}"],
                   capture_output=True)


def run_env(root, dsh_home, api_key, sandbox=None):
    node, _ = paths(root)
    sandbox = sandbox or ensure_sandbox(root)
    return dict(os.environ, DSH_HOME=str(dsh_home), COLAB_LLM_API_KEY=api_key,
                PATH=f"{sandbox}:{node.parent}:{os.environ.get('PATH', '')}")


def start(root, dsh_home, workspace, patch, api_key, port=PORT, public_url=None, trusted_hosts=(),
          log_path="/content/dsh_web.log", wait=120, sleep=time.sleep, popen=subprocess.Popen):
    """Start dsh's web UI. Returns (process, token)."""
    stop(port)
    os.makedirs(dsh_home, exist_ok=True)
    os.makedirs(workspace, exist_ok=True)
    node, _ = paths(root)
    env = run_env(root, dsh_home, api_key)
    out = open(log_path, "w")
    proc = popen(web_args(root, patch, port, public_url, trusted_hosts), cwd=str(workspace), env=env,
                 stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(int(wait)):
        token = parse_token(Path(log_path).read_text(errors="replace") if os.path.exists(log_path) else "")
        if token:
            return proc, token
        if proc.poll() is not None:
            break
        sleep(1)
    raise RuntimeError("dsh web did not start:\n" + (Path(log_path).read_text(errors="replace")[-3000:]
                                                     if os.path.exists(log_path) else ""))


def ask_open(last=None, ask=input, log=print):
    """Ask whether to open dsh here (Enter keeps the last answer, else no)."""
    last = bool(last)
    while True:
        a = ask(f"Open the dsh harness in this notebook (web UI)? [y/n, Enter = {'y' if last else 'n'}]: ").strip().lower()
        if not a:
            return last
        if a in ("y", "yes", "n", "no"):
            return a.startswith("y")
        log("  Type y or n.")


def main(argv=None):            # python -m notebook_code.dsh_web <builds>  (install only, e.g. from a CPU runtime)
    builds = (argv or sys.argv[1:] or ["/content/drive/MyDrive/colab_model_server/engines"])[0]
    print(ensure(builds))


if __name__ == "__main__":
    main()
