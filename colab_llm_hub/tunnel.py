"""Remote access to the Colab hub: Tailscale (default) or a Cloudflare tunnel.

- **Tailscale**: the Colab VM joins your tailnet with an auth key (userspace networking: Colab has no
  ``/dev/net/tun``), then ``tailscale serve`` proxies the hub over HTTPS on its MagicDNS name, so a harness
  on any machine in the same tailnet can reach it. ``funnel=True`` makes it public instead.
- **Cloudflare quick tunnel**: no account; a new public ``https://<words>.trycloudflare.com`` URL each
  session (the approach of Unsloth Studio's Colab notebook). Anyone with the URL reaches the hub, so the
  hub's API key is what protects it.
- **Cloudflare named tunnel**: a tunnel created in the Cloudflare dashboard, run with its token; the stable
  public hostname is configured there.

Binaries are cached under ``Drive/builds/tools/`` when ``drive_dir`` is set.
"""
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

# Static combined binary works without apt root installs; userspace mode needs no TUN device.
TAILSCALE_STATIC_URL = "https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz"
TS_URL_RE = re.compile(r"https://[a-z0-9._-]+\.ts\.net(?:/[^\s]*)?", re.I)


# ---- Cloudflare ---------------------------------------------------------------------------

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
# Exclude cloudflared's own api.trycloudflare.com, which appears in its failure messages.
QUICK_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", re.I)
# The quick-tunnel URL answers Cloudflare error 1033 until an edge connection registers.
REGISTERED = "Registered tunnel connection"
DOH_URL = "https://cloudflare-dns.com/dns-query?name={host}&type=A"


def ensure_cloudflared(drive_dir=None, fetch=urllib.request.urlretrieve, attempts=3, sleep=time.sleep):
    """Path to cloudflared, cached on Drive (builds/tools/cloudflared) or /content when no Drive."""
    found = shutil.which("cloudflared")
    if found:
        return found
    path = (os.path.join(drive_dir, "builds", "tools", "cloudflared") if drive_dir
            else os.environ.get("COLAB_LLM_CLOUDFLARED", "/content/cloudflared"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        os.chmod(path, 0o755)
        return path
    for attempt in range(1, attempts + 1):
        try:
            fetch(CLOUDFLARED_URL, path + ".part")
            os.replace(path + ".part", path)
            os.chmod(path, 0o755)
            return path
        except OSError:
            if attempt == attempts:
                raise
            sleep(1.5 * attempt)
    return path


def parse_quick_url(text):
    m = QUICK_RE.search(text or "")
    return m.group(0).lower() if m else None


def _read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def dns_ready(host, opener=urllib.request.urlopen):
    """True once public DNS (Cloudflare's DoH) has an A record for host. Asking DoH rather than the local
    resolver avoids caching a "no such host" answer from before the record existed."""
    req = urllib.request.Request(DOH_URL.format(host=host), headers={"accept": "application/dns-json"})
    with opener(req, timeout=5) as r:
        return any(a.get("type") == 1 for a in json.load(r).get("Answer") or [])


def probe_hub(url, api_key=None, opener=urllib.request.urlopen):
    """True when the server behind the tunnel answers /health. Cloudflare's own errors (1033 while the
    tunnel registers, 502 / 530 when the origin is unreachable) are 5xx, so a 200 (or the server's 401 to a
    missing key) means the request reached it. Works for the hub, llama-server and vLLM alike."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with opener(urllib.request.Request(url + "/health", headers=headers), timeout=5) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 401 and not api_key
    except (OSError, ValueError):
        return False


def _verify_public(url, api_key, deadline, sleep, dns_check, probe, log):
    host = url.split("//", 1)[1]
    sleep(3)                                                   # let the record exist before the first lookup
    while time.monotonic() < deadline:
        try:
            if dns_check(host):
                break
        except (OSError, ValueError):
            break                                              # DoH blocked: go straight to probing
        sleep(2)
    while time.monotonic() < deadline:
        if probe(url, api_key):
            return True
        sleep(1)
    log(f"[tunnel] {url} did not answer from here yet; it usually works within a minute from other networks.")
    return False


def start_cloudflare_quick(port, drive_dir=None, api_key=None, protocol=None, binary=None,
                           popen=subprocess.Popen, sleep=time.sleep, url_wait=30, verify_wait=45,
                           dns_check=dns_ready, probe=probe_hub, log=print):
    """Open a Cloudflare quick tunnel to the hub. Returns (process, https base URL).

    Like Unsloth Studio's Colab start(): waits until the tunnel has registered with Cloudflare's edge and
    the hub answers through it before returning the URL. If no URL appears it retries once over HTTP/2,
    for networks that block cloudflared's default QUIC."""
    binary = binary or ensure_cloudflared(drive_dir=drive_dir)
    log_path = (os.path.join(drive_dir, "logs", "cloudflared.log") if drive_dir else "/content/cloudflared.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    stop_cloudflare()
    tail = ""
    for proto in ([protocol] if protocol else [None, "http2"]):
        cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
        if proto:
            cmd += ["--protocol", proto]
        with open(log_path, "w") as out:
            proc = popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + url_wait
        url = None
        while time.monotonic() < deadline and proc.poll() is None:
            text = _read(log_path)
            url = url or parse_quick_url(text)
            if url and REGISTERED in text:
                break
            sleep(1)
        tail = _read(log_path)[-2000:]
        if url and REGISTERED in _read(log_path):
            _verify_public(url, api_key, time.monotonic() + verify_wait, sleep, dns_check, probe, log)
            return proc, url
        proc.terminate()
        log(f"[tunnel] cloudflared{' (' + proto + ')' if proto else ''} gave no working URL; "
            + ("retrying over HTTP/2" if not proto and not protocol else "giving up"))
    raise RuntimeError("Cloudflare quick tunnel did not start (trycloudflare.com may be unreachable or "
                       "rate-limited; use Tailscale instead):\n" + tail)


def start_cloudflare_named(port, token, hostname, drive_dir=None, api_key=None, binary=None,
                           popen=subprocess.Popen, sleep=time.sleep, wait=60, probe=probe_hub, log=print):
    """Run a dashboard-created tunnel by token. Its public hostname must route to http://127.0.0.1:<port>
    (set in the Cloudflare dashboard). Returns (process, https://hostname)."""
    if not token:
        raise ValueError("a Cloudflare tunnel token is required for a named tunnel")
    if not hostname:
        raise ValueError("set the tunnel's public hostname (cf_hostname), e.g. llm.example.com")
    binary = binary or ensure_cloudflared(drive_dir=drive_dir)
    log_path = (os.path.join(drive_dir, "logs", "cloudflared.log") if drive_dir else "/content/cloudflared.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    stop_cloudflare()
    with open(log_path, "w") as out:
        proc = popen([binary, "tunnel", "--no-autoupdate", "run", "--token", token],
                     stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    url = "https://" + hostname.removeprefix("https://").strip("/")
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("cloudflared exited:\n" + _read(log_path)[-2000:])
        if REGISTERED in _read(log_path) and probe(url, api_key):
            return proc, url
        sleep(1)
    log(f"[tunnel] {url} is not answering yet: check that its public hostname routes to "
        f"http://127.0.0.1:{port} in the Cloudflare dashboard.")
    return proc, url


def load_or_ask_secret(path, prompt, refresh=False, env_name=None, check=None, log=print):
    """A secret kept on private Drive: read it, or prompt once (getpass) and save it there (mode 600)."""
    from . import keyfile
    if not refresh and path:
        for cand in (path + ".txt", path):           # the .txt copy is the one a phone can open and edit
            value = keyfile.read_secret(cand, min_len=1)
            if value:
                keyfile.write_both(path, value)
                return value
    from getpass import getpass
    value = getpass(prompt).strip()
    if not value and env_name:
        try:
            from google.colab import userdata
            value = (userdata.get(env_name) or "").strip()
        except Exception:
            value = os.environ.get(env_name, "").strip()
    if not value or (check and not check(value)):
        raise ValueError("no valid value given")
    if path:
        keyfile.write_both(path, value)
        log(f"Saved to {path}.txt (private Drive, never in git)")
    return value


def stop_cloudflare():
    """Stop cloudflared tunnels started from this runtime. Anchored to the program name: an unanchored
    pattern would also match any process whose arguments merely mention cloudflared (a shell, a test)."""
    subprocess.run(["pkill", "-TERM", "-f", r"^[^ ]*cloudflared tunnel "], capture_output=True)


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


TUNNELS = (
    ("tailscale", "Tailscale: private to your tailnet, same https://colab-<engine>.<tailnet>.ts.net URL every session"),
    ("cloudflare", "Cloudflare quick tunnel: no account or key; public https://<words>.trycloudflare.com URL, "
                   "new every session (the API key still protects it)"),
    ("cloudflare-named", "Cloudflare named tunnel: your own tunnel token and a stable public hostname"),
)


def choose_tunnel(last=None, ask=input, log=print, choices=None, remember=None):
    """Ask which tunnel to open (a prompt in the cell's output, not a code edit).

    Returns (tunnel, new_key): Enter keeps `last` (or the file `remember`, else Tailscale); a number or name
    picks one; adding "new" (e.g. "1 new") asks for a new key / token even when one is saved on Drive.
    `choices` limits the menu; `remember` is a file the choice is kept in for next session."""
    options = [(n, a) for n, a in TUNNELS if not choices or n in choices]
    names = [n for n, _ in options]
    if last is None and remember:
        try:
            last = open(remember).read().strip()
        except OSError:
            last = None
    last = last if last in names else names[0]
    log("How should your machine reach this runtime?")
    for i, (name, about) in enumerate(options, 1):
        log(f"  {i}. {about}" + ("   <- last used" if name == last else ""))
    while True:
        words = ask(f"Tunnel [1-{len(options)}, Enter = {names.index(last) + 1}; add 'new' to replace a saved key]: "
                    ).strip().lower().split()
        new_key = "new" in words
        words = [w for w in words if w != "new"]
        pick = last if not words else (names[int(words[0]) - 1] if words[0].isdigit()
                                       and 1 <= int(words[0]) <= len(names) else
                                       words[0] if words[0] in names else None)
        if pick:
            if remember:
                try:
                    os.makedirs(os.path.dirname(remember), exist_ok=True)
                    with open(remember, "w") as f:
                        f.write(pick)
                except OSError:
                    pass
            return pick, new_key
        log(f"  {words[0]!r} is not one of the choices; type a number from 1 to {len(names)}.")


def ask_hostname(saved=None, ask=input, log=print):
    """The public hostname of a Cloudflare named tunnel (Enter keeps the saved one)."""
    while True:
        typed = ask("Hostname of your Cloudflare tunnel (e.g. llm.example.com)"
                    + (f" [Enter = {saved}]" if saved else "") + ": ").strip()
        value = typed or (saved or "")
        value = value.replace("https://", "").replace("http://", "").strip("/")
        if value and "." in value and " " not in value:
            return value
        log("  Type the hostname you routed to this tunnel in Cloudflare (Public Hostname).")


def load_or_ask_tailscale_authkey(drive_path=None, refresh=False, log=print):
    """Load the auth key from private Google Drive, or prompt once and save it there.

    The key lives only under the user's Drive (e.g. ``MyDrive/llm_models/hub/tailscale_authkey``).
    It is never written into notebook settings JSON or the public git repo.
    """
    from . import keyfile
    if drive_path and not refresh:
        # A key uploaded through Drive's website usually arrives as tailscale_authkey.txt; accept either.
        # The .txt copy first: it is the one Drive's phone and web apps open, so an edit there wins.
        for path in (drive_path + ".txt", drive_path):
            existing = keyfile.read_secret(path, prefix="tskey-", min_len=16)
            if existing:
                keyfile.write_both(drive_path, existing)
                log(f"Using Tailscale auth key from private Drive: {drive_path}.txt")
                return existing
    key = ask_tailscale_authkey()
    if not key.startswith("tskey-"):
        raise ValueError("Tailscale auth key must look like tskey-auth-… / tskey-…")
    if drive_path:
        keyfile.write_both(drive_path, key)
        log(f"Saved Tailscale auth key to private Drive (not in git): {drive_path}.txt")
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
    code, out = _serve(ts, socket_path, [mode, "--bg", "--yes", f"http://127.0.0.1:{port}"])
    if code and "flag provided but not defined" in out:          # older CLIs have no --yes
        code, out = _serve(ts, socket_path, [mode, "--bg", f"http://127.0.0.1:{port}"])
    if code != 0:
        proc.terminate()
        raise RuntimeError(serve_error(mode, out))
    serve = subprocess.CompletedProcess([], 0, out, "")

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


def _serve(ts, socket_path, args, timeout=45, run=subprocess.run):
    """Run tailscale serve / funnel. Returns (returncode or None on timeout, combined output). When Serve or
    Funnel is not enabled on the tailnet, the CLI prints an enable link and then waits indefinitely."""
    try:
        r = run([ts, "--socket", socket_path] + list(args), capture_output=True, text=True, timeout=timeout,
                env=_ts_env(socket_path))
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        parts = [e.stdout, e.stderr]
        return None, "".join(p.decode(errors="replace") if isinstance(p, bytes) else (p or "") for p in parts)


def serve_error(mode, output):
    m = re.search(r"https://login\.tailscale\.com/f/\S+", output or "")
    if m and "not enabled" in output:
        what = "Funnel" if mode == "funnel" else "Serve (HTTPS certificates)"
        return (f"Tailscale {what} is not enabled on your tailnet. This is a one-time admin setting: open "
                f"{m.group(0)} , enable it, then re-run this cell. (Or re-run this cell and pick 2, the Cloudflare quick tunnel.)")
    return f"tailscale {mode} failed or timed out:\n{output}"


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
