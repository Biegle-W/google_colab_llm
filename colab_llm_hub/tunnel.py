"""Public access through Cloudflare: a free quick tunnel (random URL each session) or a named tunnel (stable hostname).

The cloudflared binary is cached on Google Drive (``builds/tools/cloudflared``) so later Colab sessions
do not re-download it.
"""
import os
import re
import subprocess
import time
import urllib.request

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
QUICK_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


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
