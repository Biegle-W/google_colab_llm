"""Secrets kept on the user's private Google Drive (never committed to git).

- ``api_key`` — hub Bearer token (auto-created once)
- ``tailscale_authkey`` — Tailscale auth key (prompted once, then reused from Drive)
"""
import os
import secrets


def _chmod_private(path):
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_or_create(path):
    if os.path.exists(path):
        key = open(path).read().strip()
        if len(key) >= 24:
            return key
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    key = secrets.token_urlsafe(32)
    with open(path, "w") as f:
        f.write(key + "\n")
    _chmod_private(path)
    return key


def write_secret(path, value):
    """Write a secret file on Drive with restrictive permissions. Returns the stripped value."""
    value = (value or "").strip()
    if not value:
        raise ValueError(f"refusing to write empty secret to {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(value + "\n")
    _chmod_private(path)
    return value


def read_secret(path, prefix=None, min_len=8):
    """Return a secret from Drive, or None if missing / invalid."""
    if not path or not os.path.exists(path):
        return None
    try:
        value = open(path).read().strip()
    except OSError:
        return None
    if len(value) < min_len:
        return None
    if prefix and not value.startswith(prefix):
        return None
    return value
