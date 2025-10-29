from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from datetime import datetime

app = FastAPI(title="UAV IDS Collector")

class Obs(BaseModel):
    seq: int
    delay: float
    loss: float
    rate: float
    up_bytes: int
    down_bytes: int
    run_id: str = "default"

STORE: List[Obs] = []

@app.post("/ingest")
async def ingest(obs: Obs):
    STORE.append(obs)
    return {"status": "ok", "seq": obs.seq}

@app.get("/obs/latest")
async def latest(k: int = Query(1, ge=1, le=100)):
    return STORE[-k:]

@app.get("/obs/seq")
async def seq(since_seq: int, limit: int = Query(100, le=1000)):
    return [o for o in STORE if o.seq >= since_seq][:limit]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
