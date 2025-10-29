# 의존성 
python3 -m pip install --user fastapi uvicorn pydantic requests

# 기존 있으면 백업
cp -a ~/collector.py ~/collector.py.bak.$(date +%s) 2>/dev/null || true

# collector.py 새로 생성 (/ingest + /ingest_extra 지원)
cat > ~/collector.py <<'PY'

class Obs(BaseModel):
run_id: str
ts: float = Field(default_factory=lambda: time.time())
up_bytes: int
down_bytes: int
delay_ms: float
loss_pct: float
rate_kbps: float
seq: Optional[int] = None

STORE_EXTRA: List[Dict[str, Any]] = []
LAST_EXTRA_BY_RUN: Dict[str, Dict[str, Any]] = {}
EXTRA_FIELDS = [
"altitude_m","groundspeed_mps","battery_pct","heading_deg","heartbeat_gap_ms",
"fix_type","eph","epv","satellites_used","jamming_indicator",
"hb_hz","status_hz","paramset_hz","cmdlong_hz","gpsint_hz",
"node_id"
]

class ExtraObs(BaseModel):
run_id: str
ts: float = Field(default_factory=lambda: time.time())
altitude_m: Optional[float] = None
groundspeed_mps: Optional[float] = None
battery_pct: Optional[float] = None
heading_deg: Optional[float] = None
heartbeat_gap_ms: Optional[float] = None
fix_type: Optional[int] = None
eph: Optional[float] = None
epv: Optional[float] = None
satellites_used: Optional[int] = None
jamming_indicator: Optional[float] = None
hb_hz: Optional[float] = None
status_hz: Optional[float] = None
paramset_hz: Optional[float] = None
cmdlong_hz: Optional[float] = None
gpsint_hz: Optional[float] = None
node_id: Optional[str] = None

@app.post("/ingest_extra")
async def ingest_extra(obs: ExtraObs):
d = to_dict(obs); STORE_EXTRA.append(d); LAST_EXTRA_BY_RUN[d["run_id"]] = d
return {"status":"ok","count":len(STORE_EXTRA)}

@app.post("/ingest")
async def ingest(obs: Obs):
rid = obs.run_id
seq = SEQ_BY_RUN.get(rid, 0) + 1; SEQ_BY_RUN[rid] = seq
rec = to_dict(obs); rec["seq"] = rec.get("seq") or seq
rec["node_id"] = NODE_ID
extra = LAST_EXTRA_BY_RUN.get(rid)
if extra:
for k in EXTRA_FIELDS:
if k not in rec: rec[k] = extra.get(k, None)
STORE.append(rec)
return {"status":"ok","seq":rec["seq"],"count":len(STORE)}

@app.get("/obs/latest")
async def latest(k: int = 1):
k = max(1, min(k, 1000)); return STORE[-k:] if len(STORE) >= k else STORE

@app.get("/obs/extra/latest")
async def extra_latest(k: int = 1):
k = max(1, min(k, 1000)); return STORE_EXTRA[-k:] if len(STORE_EXTRA) >= k else STORE_EXTRA

@app.get("/obs/seq")
async def by_seq(since_seq: int = 0, limit: int = 10, run_id: Optional[str] = None):
rid = run_id or (STORE[-1]["run_id"] if STORE else None)
if not rid: return []
out = [r for r in STORE if r["run_id"] == rid and r["seq"] >= since_seq]
return out[:max(1, min(limit, 1000))]

if __name__ == "__main__":
uvicorn.run(app, host="0.0.0.0", port=8080)
PY

# 실행 이 터미널 켜두기 
pkill -f "collector.py|uvicorn" 2>/dev/null || true
python3 ~/collector.py &
