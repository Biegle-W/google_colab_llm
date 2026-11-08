"""Keep writes into Colab's Google Drive mount from running ahead of the actual upload.

Colab's Drive client accepts writes into a local cache much faster than it uploads them. On a bulk save the
backlog grew to 48 GB, the client stopped uploading, and the runtime was lost with the data still in its
cache. A copier made by `upload_paced_copier` returns only after the runtime has sent about as many bytes
upstream as the file it just wrote (read from the network counters), so at most one file is ever waiting.
If uploads stop moving for `stall_s` it raises instead of piling up more.
"""
import time

from .store import copy_file_atomic


class UploadStalled(RuntimeError):
    pass


def tx_bytes(read=lambda: open("/proc/net/dev").read()):
    """Bytes sent on all non-loopback interfaces."""
    total = 0
    for line in read().splitlines()[2:]:
        name, _, data = line.partition(":")
        if name.strip() != "lo" and data.split():
            total += int(data.split()[8])
    return total


def wait_uploaded(nbytes, tx_start, tx=tx_bytes, stall_s=1200, poll_s=15, sleep=time.sleep, now=time.monotonic,
                  log=print, label=""):
    """Block until about `nbytes` have gone out since `tx_start`. Raises UploadStalled when nothing moves."""
    target = tx_start + int(nbytes * 0.97)         # other traffic is small; protocol overhead adds a little
    last, last_change = tx(), now()
    reported = 0
    while last < target:
        sleep(poll_s)
        cur = tx()
        if cur > last + 1_000_000:
            last, last_change = cur, now()
        elif now() - last_change > stall_s:
            raise UploadStalled(f"Drive upload stopped for {stall_s // 60} min with "
                                f"{(target - cur) / 1e9:.1f} GB of {label or 'a file'} still to send")
        done = (cur - tx_start) / max(nbytes, 1)
        if done - reported >= 0.25:
            reported = done
            log(f"[upload] {label}: {min(done, 1):.0%} sent to Drive")
    return True


def upload_paced_copier(tx=tx_bytes, stall_s=1200, sleep=time.sleep, log=print, copy=copy_file_atomic):
    """A copier for ModelStore.download_to_drive that waits for each file's upload before returning."""
    def copier(src, dst, on_bytes=None):
        import os
        size = os.path.getsize(src)
        start = tx()
        copy(src, dst, on_bytes)
        if size > 50_000_000:                       # small files ride along with the next big one
            wait_uploaded(size, start, tx=tx, stall_s=stall_s, sleep=sleep, log=log, label=os.path.basename(str(dst)))
    return copier
