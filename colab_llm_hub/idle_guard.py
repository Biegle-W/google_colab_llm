"""Release this Colab runtime after a period of inactivity, so an idle VM stops using compute units.

Colab (and the Colab CLI) keep a runtime alive while it is connected, idle or not. This guard runs as a
detached process on the VM and counts the runtime as active while any of these hold:

- a notebook / CLI cell is running, or ran within the window (the local Jupyter server's kernel activity)
- the hub is answering, loading or downloading (it touches HEARTBEAT; see hub.py)
- CPU, network or GPU are busy (builds, downloads, Drive uploads, inference)

After `minutes` with none of them, it finishes pending Google Drive uploads (drive.flush_and_unmount, so
nothing written to Drive is lost) and asks Colab to unassign the VM: the same call as
google.colab.runtime.unassign(), which posts to the runtime management service at $TBE_RUNTIME_ADDR.

    python -m colab_llm_hub.idle_guard --minutes 10        # or idle_guard.start(10) from a notebook
"""
import argparse
import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HEARTBEAT = "/tmp/colab-llm-hub.activity"
PIDFILE = "/tmp/colab-llm-idle-guard.pid"
LOG = "/content/idle_guard.log"
KERNELS_URL = "http://127.0.0.1:8080/api/kernels"
CPU_BUSY_CORES = 0.5          # sustained use above half a core (builds, downloads, engines working)
NET_BUSY_BYTES_S = 500_000    # downloads, Drive uploads, streamed replies
GPU_BUSY_PERCENT = 5


def touch_activity(path=HEARTBEAT):
    """Called by the hub on requests, loads and downloads."""
    try:
        with open(path, "a"):
            os.utime(path, None)
    except OSError:
        pass


def _iso(ts):
    return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def kernel_activity(url=KERNELS_URL, opener=urllib.request.urlopen):
    """(any kernel busy, latest kernel activity as epoch seconds), or (False, None) if unavailable."""
    try:
        with opener(url, timeout=5) as r:
            kernels = json.load(r)
    except (OSError, ValueError):
        return False, None
    busy = any(k.get("execution_state") == "busy" for k in kernels)
    times = [_iso(k["last_activity"]) for k in kernels if k.get("last_activity")]
    return busy, (max(times) if times else None)


def cpu_jiffies(read=lambda: open("/proc/stat").readline()):
    parts = [int(x) for x in read().split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
    return sum(parts), idle


def net_bytes(read=lambda: open("/proc/net/dev").read()):
    total = 0
    for line in read().splitlines()[2:]:
        name, _, data = line.partition(":")
        if name.strip() == "lo":
            continue
        fields = data.split()
        total += int(fields[0]) + int(fields[8])
    return total


def gpu_percent(run=subprocess.run):
    try:
        r = run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
        return max(int(x) for x in r.stdout.split()) if r.returncode == 0 and r.stdout.strip() else 0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


class Guard:
    def __init__(self, minutes, interval=30, now=time.time, sleep=time.sleep, log=print,
                 kernels=kernel_activity, cpu=cpu_jiffies, net=net_bytes, gpu=gpu_percent,
                 heartbeat=HEARTBEAT, release=None, ncpu=None):
        self.timeout = minutes * 60
        self.interval, self.now, self.sleep, self.log = interval, now, sleep, log
        self.kernels, self.cpu, self.net, self.gpu, self.heartbeat = kernels, cpu, net, gpu, heartbeat
        self.release = release or release_runtime
        self.ncpu = ncpu or os.cpu_count() or 1
        self.last_active = now()
        self._cpu, self._net, self._t = cpu(), net(), now()

    def reasons(self):
        """Why the runtime counts as active right now (empty list = idle)."""
        out = []
        t = self.now()
        dt = max(t - self._t, 1e-6)
        cpu, net = self.cpu(), self.net()
        total, idle = cpu[0] - self._cpu[0], cpu[1] - self._cpu[1]
        busy_cores = (1 - idle / total) * self.ncpu if total > 0 else 0.0
        net_rate = (net - self._net) / dt
        self._cpu, self._net, self._t = cpu, net, t
        if busy_cores > CPU_BUSY_CORES:
            out.append(f"cpu {busy_cores:.1f} cores")
        if net_rate > NET_BUSY_BYTES_S:
            out.append(f"network {net_rate / 1e6:.1f} MB/s")
        if self.gpu() > GPU_BUSY_PERCENT:
            out.append("gpu busy")
        busy, last = self.kernels()
        if busy:
            out.append("cell running")
        try:
            hb = os.path.getmtime(self.heartbeat)
        except OSError:
            hb = None
        recent = max([x for x in (last, hb) if x] or [0])
        if recent > self.last_active:
            self.last_active = min(recent, t)
        return out

    def step(self):
        """One check. Returns True when the runtime was released."""
        if self.reasons():
            self.last_active = self.now()
            return False
        idle = self.now() - self.last_active
        if idle >= self.timeout:
            self.log(f"[idle-guard] idle for {idle / 60:.0f} min: releasing the runtime")
            return self.release(self.log)
        return False

    def run(self):
        self.log(f"[idle-guard] releasing this runtime after {self.timeout // 60:.0f} min of inactivity")
        while True:
            self.sleep(self.interval)
            try:
                if self.step():
                    return
            except Exception as e:                  # never die quietly: a dead guard means a VM left running
                self.log(f"[idle-guard] check failed: {e}")


def flush_drive(log=print):
    """Finish pending Drive uploads and unmount (no-op when Drive is not mounted). False on failure."""
    try:
        if b"fuse.drive" not in subprocess.check_output(["/bin/mount"]):
            return True
        from google.colab import drive
        drive.flush_and_unmount(timeout_ms=30 * 60 * 1000)
        log("[idle-guard] Drive uploads finished")
        return True
    except Exception as e:
        log(f"[idle-guard] could not flush Drive ({e}); not releasing yet")
        return False


def release_runtime(log=print, flush=flush_drive, opener=urllib.request.urlopen):
    if not flush(log):
        return False
    addr = os.environ.get("TBE_RUNTIME_ADDR")
    if not addr:
        log("[idle-guard] not on Colab (no TBE_RUNTIME_ADDR); nothing to release")
        return True
    req = urllib.request.Request(f"http://{addr}/unassign", data=b"", method="POST")
    with opener(req, timeout=30) as r:
        log(f"[idle-guard] unassign requested: HTTP {r.status}")
    return True


def start(minutes=10, log_path=LOG, python=None):
    """Start (or replace) the detached guard. minutes <= 0 stops it. Returns the guard's pid or None."""
    stop()
    if not minutes or minutes <= 0:
        return None
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=pkg_parent + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with open(log_path, "a") as out:
        proc = subprocess.Popen([python or sys.executable, "-m", "colab_llm_hub.idle_guard", "--minutes", str(minutes)],
                                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env=env)
    with open(PIDFILE, "w") as f:
        f.write(str(proc.pid))
    return proc.pid


def stop():
    try:
        pid = int(open(PIDFILE).read().strip())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        pass
    try:
        os.remove(PIDFILE)
    except OSError:
        pass


def main(argv=None):
    p = argparse.ArgumentParser(prog="colab_llm_hub.idle_guard")
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--interval", type=float, default=30)
    a = p.parse_args(argv)
    Guard(a.minutes, a.interval, log=lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)).run()


if __name__ == "__main__":
    sys.exit(main())
