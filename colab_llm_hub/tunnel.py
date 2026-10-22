"""Public access through Cloudflare: a free quick tunnel (random URL each session) or a named tunnel (stable hostname)."""
import os
import re
import subprocess
import time
import urllib.request

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
QUICK_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def ensure_cloudflared(path="/content/cloudflared", fetch=urllib.request.urlretrieve):
    if not os.path.exists(path):
        fetch(CLOUDFLARED_URL, path)
        os.chmod(path, 0o755)
    return path


def parse_quick_url(text):
    m = QUICK_RE.search(text or "")
    return m.group(0) if m else None


def start_quick(port, binary="/content/cloudflared", log_path="/content/cloudflared.log", popen=subprocess.Popen,
                wait=60, sleep=time.sleep):
    """Open a quick tunnel to 127.0.0.1:port. Returns (process, public URL)."""
    with open(log_path, "w") as log:
        proc = popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                     stdout=log, stderr=subprocess.STDOUT)
    for _ in range(wait):
        url = parse_quick_url(open(log_path).read())
        if url:
            return proc, url
        sleep(1)
    raise RuntimeError("No tunnel URL yet:\n" + open(log_path).read()[-2000:])


def start_named(token, binary="/content/cloudflared", log_path="/content/cloudflared.log", popen=subprocess.Popen):
    """Run a named tunnel created in the Cloudflare dashboard; its public hostname is configured there."""
    with open(log_path, "w") as log:
        return popen([binary, "tunnel", "--no-autoupdate", "run", "--token", token],
                     stdout=log, stderr=subprocess.STDOUT)
