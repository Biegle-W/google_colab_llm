"""A stable API key kept on Drive, so the local harness only needs configuring once."""
import os
import secrets


def load_or_create(path):
    if os.path.exists(path):
        key = open(path).read().strip()
        if len(key) >= 24:
            return key
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = secrets.token_urlsafe(32)
    with open(path, "w") as f:
        f.write(key + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key
