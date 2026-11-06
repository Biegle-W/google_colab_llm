"""Public / private access tunnels for the Colab hub.

- **Tailscale** (default): joins your tailnet with an auth key, then ``tailscale serve``
  proxies the hub. Requires userspace networking on Colab (no ``/dev/net/tun``).
- **Cloudflare quick**: free random ``*.trycloudflare.com`` URL each session.
- **Cloudflare named**: stable hostname via a dashboard tunnel token.

Binaries are cached under ``Drive/builds/tools/`` when ``drive_dir`` is set.
"""
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
# Static combined binary works without apt root installs; userspace mode needs no TUN device.
TAILSCALE_STATIC_URL = "https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz"
QUICK_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
TS_URL_RE = re.compile(r"https://[a-z0-9._-]+\.ts\.net(?:/[^\s]*)?", re.I)


def ensure_cloudflared(path=None, fetch=urllib.request.urlretrieve, drive_dir=None):
    if path is None:
        if drive_dir:
            path = os.path.join(drive_dir, "builds", "tools", "cloudflared")
        else:
            path = os.environ.get("COLAB_LLM_CLOUDFLARED", "/content/cloudflared")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        part = path + ".part"
        fetch(CLOUDFLARED_URL, part)
        os.replace(part, path)
        os.chmod(path, 0o755)
    elif not os.access(path, os.X_OK):
        os.chmod(path, 0o755)
    return path


def parse_quick_url(text):
    m = QUICK_RE.search(text or "")
    return m.group(0) if m else None


def start_quick(port, binary=None, log_path=None, popen=subprocess.Popen,
                wait=60, sleep=time.sleep, drive_dir=None):
    """Open a quick tunnel to 127.0.0.1:port. Returns (process, public URL)."""
    binary = binary or ensure_cloudflared(drive_dir=drive_dir)
    if log_path is None:
        log_path = (os.path.join(drive_dir, "logs", "cloudflared.log") if drive_dir
                    else "/content/cloudflared.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as log:
        proc = popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                     stdout=log, stderr=subprocess.STDOUT)
    for _ in range(wait):
        url = parse_quick_url(open(log_path).read())
        if url:
            return proc, url
        sleep(1)
    raise RuntimeError("No tunnel URL yet:\n" + open(log_path).read()[-2000:])


def start_named(token, binary=None, log_path=None, popen=subprocess.Popen, drive_dir=None):
    """Run a named tunnel created in the Cloudflare dashboard; its public hostname is configured there."""
    binary = binary or ensure_cloudflared(drive_dir=drive_dir)
    if log_path is None:
        log_path = (os.path.join(drive_dir, "logs", "cloudflared.log") if drive_dir
                    else "/content/cloudflared.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as log:
        return popen([binary, "tunnel", "--no-autoupdate", "run", "--token", token],
                     stdout=log, stderr=subprocess.STDOUT)


# ---- Tailscale ---------------------------------------------------------------------------

def ensure_tailscale(drive_dir=None, fetch=urllib.request.urlretrieve, run=subprocess.run):
    """Return (tailscale, tailscaled) binary paths, restoring/caching under Drive when possible."""
    tools = (os.path.join(drive_dir, "builds", "tools", "tailscale") if drive_dir
             else "/content/tailscale")
    os.makedirs(tools, exist_ok=True)
    ts = os.path.join(tools, "tailscale")
    tsd = os.path.join(tools, "tailscaled")
    if os.path.isfile(ts) and os.path.isfile(tsd) and os.access(ts, os.X_OK) and os.access(tsd, os.X_OK):
        return ts, tsd
    # Prefer a system install when present (apt / install.sh).
    which_ts = shutil.which("tailscale")
    which_tsd = shutil.which("tailscaled")
    if which_ts and which_tsd:
        return which_ts, which_tsd
    tarball = os.path.join(tools, "tailscale_latest_amd64.tgz")
    fetch(TAILSCALE_STATIC_URL, tarball)
    run(["tar", "-xzf", tarball, "-C", tools], check=True, capture_output=True)
    # Archive extracts to tailscale_X.Y.Z_amd64/{tailscale,tailscaled}
    found_ts = found_tsd = None
    for dirpath, _, files in os.walk(tools):
        if "tailscale" in files and "tailscaled" in files:
            found_ts = os.path.join(dirpath, "tailscale")
            found_tsd = os.path.join(dirpath, "tailscaled")
            break
    if not (found_ts and found_tsd):
        raise RuntimeError(f"Tailscale archive did not contain binaries under {tools}")
    # Keep stable paths at tools/tailscale and tools/tailscaled.
    if found_ts != ts:
        shutil.copy2(found_ts, ts)
        shutil.copy2(found_tsd, tsd)
    os.chmod(ts, 0o755)
    os.chmod(tsd, 0o755)
    return ts, tsd


def parse_tailscale_url(text):
    """Pick the first https://….ts.net URL from serve/status output."""
    m = TS_URL_RE.search(text or "")
    return m.group(0).rstrip("/") if m else None


def _ts_env(socket_path):
    env = dict(os.environ)
    env["TS_SOCKET"] = socket_path
    return env


def _run_ts(ts_bin, socket_path, args, check=True, timeout=120):
    cmd = [ts_bin, "--socket", socket_path] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check,
                          env=_ts_env(socket_path))


def ask_tailscale_authkey(prompt=None):
    """Prompt for a Tailscale auth key (getpass). Empty Enter may use Colab secret."""
    from getpass import getpass
    prompt = prompt or (
        "Tailscale auth key (tskey-auth-…; reusable+ephemeral recommended).\n"
        "Create at https://login.tailscale.com/admin/settings/keys\n"
        "Paste key (Enter = use Colab secret TAILSCALE_AUTHKEY if set): "
    )
    typed = getpass(prompt).strip()
    if typed:
        return typed
    try:
        from google.colab import userdata
        secret = (userdata.get("TAILSCALE_AUTHKEY") or "").strip()
        if secret:
            print("Using Colab secret TAILSCALE_AUTHKEY")
            return secret
    except Exception:
        pass
    raise RuntimeError(
        "No Tailscale auth key provided. Generate a reusable ephemeral key at "
        "https://login.tailscale.com/admin/settings/keys and re-run this cell."
    )


def load_or_ask_tailscale_authkey(drive_path=None, refresh=False, log=print):
    """Load the auth key from private Google Drive, or prompt once and save it there.

    The key lives only under the user's Drive (e.g. ``MyDrive/llm_models/hub/tailscale_authkey``).
    It is never written into notebook settings JSON or the public git repo.
    """
    from . import keyfile
    if drive_path and not refresh:
        existing = keyfile.read_secret(drive_path, prefix="tskey-", min_len=16)
        if existing:
            log(f"Using Tailscale auth key from private Drive: {drive_path}")
            return existing
    key = ask_tailscale_authkey()
    if not key.startswith("tskey-"):
        raise ValueError("Tailscale auth key must look like tskey-auth-… / tskey-…")
    if drive_path:
        keyfile.write_secret(drive_path, key)
        log(f"Saved Tailscale auth key to private Drive (not in git): {drive_path}")
    return key


def start_tailscale(port, authkey, hostname=None, drive_dir=None, funnel=False,
                    popen=subprocess.Popen, sleep=time.sleep, wait=90):
    """Join the tailnet and Serve (or Funnel) the hub. Returns (tailscaled_proc, https_base_url).

    Uses ``--tun=userspace-networking`` and ``--state=mem:`` so Colab needs no root TUN device
    and the node disappears when the runtime dies (pair with an ephemeral auth key).
    """
    if not authkey or not str(authkey).startswith("tskey-"):
        raise ValueError("Tailscale auth key must look like tskey-auth-… / tskey-…")
    ts, tsd = ensure_tailscale(drive_dir=drive_dir)
    hostname = hostname or os.environ.get("COLAB_LLM_TS_HOSTNAME") or "colab-llm-hub"
    runtime = "/tmp/colab-llm-tailscale"
    os.makedirs(runtime, exist_ok=True)
    socket_path = os.path.join(runtime, "tailscaled.sock")
    state_dir = "mem:"  # ephemeral in-memory state for Colab sessions
    log_path = (os.path.join(drive_dir, "logs", "tailscaled.log") if drive_dir
                else "/content/tailscaled.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # Stop any previous userspace daemon from an earlier cell re-run.
    subprocess.run(["pkill", "-TERM", "-f", f"[t]ailscaled.*{re.escape(socket_path)}"],
                   capture_output=True)
    sleep(1)
    if os.path.exists(socket_path):
        try:
            os.remove(socket_path)
        except OSError:
            pass

    with open(log_path, "ab") as log:
        proc = popen(
            [tsd, f"--tun=userspace-networking", f"--state={state_dir}",
             f"--socket={socket_path}", "--statedir", runtime],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )

    # Wait for the socket, then authenticate.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if os.path.exists(socket_path) or proc.poll() is not None:
            break
        sleep(0.2)
    if proc.poll() is not None:
        raise RuntimeError("tailscaled exited early:\n" + open(log_path).read()[-2000:])

    up = _run_ts(ts, socket_path, [
        "up", "--auth-key=" + authkey, "--hostname=" + hostname,
        "--accept-dns=false", "--ssh=false", "--reset",
    ], check=False, timeout=180)
    if up.returncode:
        proc.terminate()
        raise RuntimeError(
            "tailscale up failed:\n" + (up.stdout or "") + (up.stderr or "") +
            "\n--- tailscaled.log ---\n" + open(log_path).read()[-2000:]
        )

    mode = "funnel" if funnel else "serve"
    # Proxy the hub over HTTPS on the MagicDNS name (Serve = tailnet only; Funnel = public).
    serve = _run_ts(ts, socket_path, [mode, "--bg", "--yes", f"http://127.0.0.1:{port}"],
                    check=False, timeout=120)
    if serve.returncode:
        # Older CLIs may not support --yes; retry without it.
        serve = _run_ts(ts, socket_path, [mode, "--bg", f"http://127.0.0.1:{port}"],
                        check=False, timeout=120)
    if serve.returncode:
        proc.terminate()
        raise RuntimeError(
            f"tailscale {mode} failed:\n" + (serve.stdout or "") + (serve.stderr or "")
        )

    url = None
    for _ in range(wait):
        status = _run_ts(ts, socket_path, [mode, "status"], check=False)
        url = parse_tailscale_url((status.stdout or "") + (status.stderr or ""))
        if not url:
            # Fall back to DNSName from `tailscale status --json`.
            st = _run_ts(ts, socket_path, ["status", "--json"], check=False)
            try:
                data = json.loads(st.stdout or "{}")
                dns = (data.get("Self") or {}).get("DNSName") or ""
                dns = dns.rstrip(".")
                if dns:
                    url = "https://" + dns
            except ValueError:
                pass
        if url:
            return proc, url.rstrip("/")
        sleep(1)
    raise RuntimeError(
        f"Tailscale {mode} started but no https://….ts.net URL appeared.\n"
        + (serve.stdout or "") + open(log_path).read()[-1500:]
    )


def stop_tailscale(drive_dir=None):
    """Best-effort teardown of a userspace Tailscale daemon started by this hub."""
    runtime = "/tmp/colab-llm-tailscale"
    socket_path = os.path.join(runtime, "tailscaled.sock")
    try:
        ts, _ = ensure_tailscale(drive_dir=drive_dir)
        if os.path.exists(socket_path):
            _run_ts(ts, socket_path, ["down"], check=False, timeout=30)
            _run_ts(ts, socket_path, ["logout"], check=False, timeout=30)
    except Exception:
        pass
    subprocess.run(["pkill", "-TERM", "-f", "[t]ailscaled.*colab-llm-tailscale"],
                   capture_output=True)
