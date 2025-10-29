import socket
import select
import time
import threading
import subprocess
import os
import random

# Constants
QGC_LISTEN_PORT = 14640  # QGC "Connect to host"
PX4_HOST = "127.0.0.1"
PX4_CMD_PORT = 14540  # PX4 Commands (Uplink)
PX4_TEL_PORT = 14550  # PX4 Telemetry (Downlink)
NS3_PATH = os.environ.get("NS3_PATH", os.path.expanduser("~/ns-allinone-3.35/ns-3.35"))
UPDATE_HZ = 1.0  # Metrics update rate

DEFAULT_DELAY_MS = 20.0
DEFAULT_LOSS_PCT = 0.0
DEFAULT_RATE_KBPS = 5000.0

# Shared metrics (thread-safe)
_metrics = {"delay_ms": DEFAULT_DELAY_MS, "loss_pct": DEFAULT_LOSS_PCT, "rate_kbps": DEFAULT_RATE_KBPS}
metrics_lock = threading.Lock()

def set_metrics(delay=None, loss=None, rate=None):
    with metrics_lock:
        if delay is not None: _metrics["delay_ms"] = float(delay)
        if loss is not None: _metrics["loss_pct"] = float(loss)
        if rate is not None: _metrics["rate_kbps"] = max(1.0, float(rate))

def get_metrics():
    with metrics_lock:
        return dict(_metrics)

class TokenBucket:
    def __init__(self, kbps):
        self.kbps = max(1.0, float(kbps))
        self.tokens = 0.0
        self.t = time.time()

    def update(self, kbps):
        self.kbps = max(1.0, float(kbps))

    def pace(self, nbytes):
        now = time.time()
        dt = now - self.t
        self.t = now
        self.tokens += (self.kbps * 1000.0 / 8.0) * dt  # bytes per sec
        need = float(nbytes)
        if self.tokens >= need:
            self.tokens -= need
            return
        wait_s = (need - self.tokens) / (self.kbps * 1000.0 / 8.0)
        self.tokens = 0.0
        if wait_s > 0:
            time.sleep(wait_s)

tb_up = TokenBucket(DEFAULT_RATE_KBPS)
tb_down = TokenBucket(DEFAULT_RATE_KBPS)

def maybe_drop(loss_pct):
    return random.random() < (float(loss_pct) / 100.0)

def shape_and_send(sock_out, data, dst, direction="up"):
    m = get_metrics()
    if maybe_drop(m["loss_pct"]):
        return  # Drop
    if m["delay_ms"] > 0:
        time.sleep(m["delay_ms"] / 1000.0)
    if direction == "up":
        tb_up.pace(len(data))
    else:
        tb_down.pace(len(data))
    sock_out.sendto(data, dst)

def ns3_metrics_loop():
    # NS-3 program: scratch/mw-link-metrics (from positions.txt)
    cmd = f"./ns3 run 'scratch/mw-link-metrics'"
    while True:
        try:
            r = subprocess.run(cmd, cwd=NS3_PATH, shell=True, text=True, capture_output=True)
            d = l = rt = None
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if line.startswith("DELAY_MS:"):
                        d = float(line.split(":")[1])
                    elif line.startswith("LOSS_PCT:"):
                        l = float(line.split(":")[1])
                    elif line.startswith("RATE_KBPS:"):
                        rt = float(line.split(":")[1])
                set_metrics(d, l, rt)
                if rt is not None:
                    tb_up.update(rt)
                    tb_down.update(rt)
        except Exception:
            pass  # NS-3 error: continue
        time.sleep(1.0 / UPDATE_HZ)

def main():
    # QGC Listen (Uplink from QGC)
    sock_qgc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_qgc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_qgc.bind(("0.0.0.0", QGC_LISTEN_PORT))

    # PX4 Tel Listen (Downlink from PX4)
    sock_px4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_px4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_px4.bind(("0.0.0.0", PX4_TEL_PORT))

    # Outbound sockets
    sock_to_px4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_to_qgc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    last_qgc_addr = None

    print(f"[MW] Up: QGC->{QGC_LISTEN_PORT} | PX4-MW:{PX4_TEL_PORT} -> PX4 ({PX4_HOST})")

    # Start NS-3 metrics thread
    threading.Thread(target=ns3_metrics_loop, daemon=True).start()

    upB = downB = 0
    t0 = time.time()

    while True:
        r, _, _ = select.select([sock_qgc, sock_px4], [], [], 1.0)

        # Uplink: QGC -> PX4
        if sock_qgc in r:
            data, addr = sock_qgc.recvfrom(65535)
            if last_qgc_addr is None:
                last_qgc_addr = addr  # Capture first QGC addr
            shape_and_send(sock_to_px4, data, (PX4_HOST, PX4_CMD_PORT), "up")
            upB += len(data)

        # Downlink: PX4 -> QGC
        if sock_px4 in r:
            data, paddr = sock_px4.recvfrom(65535)
            if last_qgc_addr:  # QGC addr captured
                shape_and_send(sock_to_qgc, data, last_qgc_addr, "down")
                downB += len(data)

        # 1Hz Log
        if time.time() - t0 > 1.0:
            m = get_metrics()
            print(f"[MW] up:{upB}B down:{downB}B | delay={m['delay_ms']:.1f}ms loss={m['loss_pct']:.1f}%")
            upB = downB = 0
            t0 = time.time()

if __name__ == "__main__":
    main()
