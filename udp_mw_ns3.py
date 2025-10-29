import socket
import time
import random
import threading
import requests  # HTTP POST
import json

# Config
UPLINK_IN = 14640   # QGC -> MW
UPLINK_OUT = 14540  # MW -> PX4
DOWNLINK_IN = 14550 # PX4 -> MW
DOWNLINK_OUT = 0    # Dynamic QGC

IDS_ENDPOINT = "http://localhost:8080/ingest"  # .env or export

delay, loss, rate = 20, 0.0, 5000  # Defaults
seq = 0

def ns3_listener():
    global delay, loss, rate
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 1550))  # NS-3 UDP
    while True:
        data, _ = sock.recvfrom(1024)
        metrics = json.loads(data)
        delay, loss, rate = metrics['delay'], metrics['loss'], metrics['rate']

def shape_packet(data: bytes, is_up: bool) -> bytes:
    global seq
    if random.random() < loss / 100:
        return b''  # Drop
    time.sleep(delay / 1000)
    seq += 1
    # POST to collector
    obs = {"seq": seq, "delay": delay, "loss": loss, "rate": rate,
           "up_bytes": len(data) if is_up else 0, "down_bytes": len(data) if not is_up else 0}
    requests.post(IDS_ENDPOINT, json=obs.dict())
    return data

# Uplink Thread: QGC -> PX4
def uplink():
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in.bind(('127.0.0.1', UPLINK_IN))
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        data, addr = sock_in.recvfrom(1024)
        shaped = shape_packet(data, True)
        if shaped: sock_out.sendto(shaped, ('127.0.0.1', UPLINK_OUT))

# Downlink Thread: PX4 -> QGC
down_port = None
def downlink():
    global down_port
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in.bind(('127.0.0.1', DOWNLINK_IN))
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        data, addr = sock_in.recvfrom(4096)
        if down_port is None:
            down_port = addr[1]  # Capture QGC port
        shaped = shape_packet(data, False)
        if shaped and down_port:
            sock_out.sendto(shaped, ('127.0.0.1', down_port))

if __name__ == "__main__":
    threading.Thread(target=ns3_listener, daemon=True).start()
    threading.Thread(target=uplink, daemon=True).start()
    threading.Thread(target=downlink).start()
    print("🚀 MW Running | Ctrl+C to stop")
    try: input()
    except: pass
