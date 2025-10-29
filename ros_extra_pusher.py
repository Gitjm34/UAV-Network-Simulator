cat > ~/mw_ns3/ros_extra_pusher.py <<'PY'

import os, time, math, threading, requests, socket
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State, Mavlink, GPSRAW
from sensor_msgs.msg import NavSatFix, NavSatStatus
from collections import Counter

RUN_ID = os.environ.get("RUN_ID", "run_"+time.strftime("%y%m%d_%H%M%S"))
ENDPOINT = os.environ.get("IDS_EXTRA_ENDPOINT", "http://127.0.0.1:8080/ingest_extra")
NODE = socket.gethostname()

# --- state ---
altitude_m = None
groundspeed_mps = None
battery_pct = None
heading_deg = None
heartbeat_gap_ms = None

gnss = {"fix_type": None, "eph": None, "epv": None, "satellites_used": None, "jamming_indicator": None}

cnt = Counter(); prev = Counter(); prev_ts = time.time()
HB=0; SYS_STATUS=1; PARAM_SET=23; COMMAND_LONG=76; GLOBAL_POS_INT=33

def cb_alt(msg: Float64):
    global altitude_m; altitude_m = float(msg.data)

def cb_vel(msg: TwistStamped):
    global groundspeed_mps
    vx, vy = msg.twist.linear.x, msg.twist.linear.y
    groundspeed_mps = float(math.sqrt(vx*vx + vy*vy))

_last_hb = None
def cb_state(msg: State):
    global heartbeat_gap_ms, _last_hb
    now = msg.header.stamp.to_sec() if msg._has_header else time.time()
    if _last_hb is not None:
        heartbeat_gap_ms = float((now - _last_hb) * 1000.0)
    _last_hb = now

def cb_hdg(msg: Float64):
    global heading_deg; heading_deg = float(msg.data)

# (A) 있으면 원시 GPSRAW 사용
def cb_gpsraw(msg: GPSRAW):
    gnss["fix_type"] = int(msg.fix_type)
    gnss["eph"] = float(msg.eph) if msg.eph > 0 else None
    gnss["epv"] = float(msg.epv) if msg.epv > 0 else None
    gnss["satellites_used"] = int(msg.satellites_visible)

# (B) SITL 대체: NavSatFix로 eph/epv 근사 + fix_type 매핑
def cb_navsat(msg: NavSatFix):
    # covariance → 표준편차 근사 (m)
    if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
        try:
            gnss["eph"] = float(math.sqrt(max(0.0, msg.position_covariance[0])))
            gnss["epv"] = float(math.sqrt(max(0.0, msg.position_covariance[8])))
        except Exception:
            pass
    # fix_type 근사
    st = msg.status.status
    if st == NavSatStatus.STATUS_NO_FIX:        gnss["fix_type"] = 0
    elif st == NavSatStatus.STATUS_FIX:         gnss["fix_type"] = 3
    elif st == NavSatStatus.STATUS_SBAS_FIX:    gnss["fix_type"] = 4
    elif st == NavSatStatus.STATUS_GBAS_FIX:    gnss["fix_type"] = 5

def cb_mavlink_from(msg: Mavlink):
    cnt[int(msg.msgid)] += 1

def push_loop():
    global prev, prev_ts
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        now = time.time()
        dt = max(1e-3, now - prev_ts)
        snap = cnt.copy()
        delta = Counter({k: snap[k] - prev.get(k, 0) for k in snap})
        prev, prev_ts = snap, now

        def hz(mid): return delta.get(mid, 0)/dt
        payload = {
            "run_id": RUN_ID, "ts": now, "node_id": NODE,
            "altitude_m": altitude_m, "groundspeed_mps": groundspeed_mps,
            "battery_pct": battery_pct, "heading_deg": heading_deg,
            "heartbeat_gap_ms": heartbeat_gap_ms,
            "fix_type": gnss["fix_type"], "eph": gnss["eph"], "epv": gnss["epv"],
            "satellites_used": gnss["satellites_used"],
            "jamming_indicator": gnss["jamming_indicator"],
            "hb_hz": hz(HB), "status_hz": hz(SYS_STATUS),
            "paramset_hz": hz(PARAM_SET), "cmdlong_hz": hz(COMMAND_LONG),
            "gpsint_hz": hz(GLOBAL_POS_INT),
        }
        try: requests.post(ENDPOINT, json=payload, timeout=0.8)
        except Exception: pass
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("ros_extra_pusher")
    rospy.Subscriber("/mavros/global_position/rel_alt", Float64, cb_alt, queue_size=1)
    rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, cb_vel, queue_size=1)
    rospy.Subscriber("/mavros/state", State, cb_state, queue_size=1)
    rospy.Subscriber("/mavros/global_position/compass_hdg", Float64, cb_hdg, queue_size=1)

    # GNSS 원시 & 대체 모두 구독 (가능한 쪽이 값 채움)
    rospy.Subscriber("/mavros/gpsstatus/gps1/raw", GPSRAW, cb_gpsraw, queue_size=1)
    rospy.Subscriber("/mavros/global_position/raw/fix", NavSatFix, cb_navsat, queue_size=1)

    # MAVLink RAW (mavlink 플러그인 필요)
    rospy.Subscriber("/mavlink/from", Mavlink, cb_mavlink_from, queue_size=200)

    threading.Thread(target=push_loop, daemon=True).start()
    rospy.spin()
PY
chmod +x ~/mw_ns3/ros_extra_pusher.py
