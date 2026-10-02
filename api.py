"""Scoring API. Send a machine's recent history, get its failure risk for the next window.

Run: uvicorn api:app --reload   -> docs at http://localhost:8000/docs
"""
import datetime as dt
import json
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import model_dir
from features import FEATURES, MIN_HISTORY, build_features

app = FastAPI(title="Predictive Maintenance API", version="1.0")


class Reading(BaseModel):
    datetime: dt.datetime
    volt: float
    rotate: float
    pressure: float
    vibration: float


class ErrorEvent(BaseModel):
    datetime: dt.datetime
    errorID: str = Field(examples=["error1"])


class MaintEvent(BaseModel):
    datetime: dt.datetime
    comp: str = Field(examples=["comp2"])


class PredictRequest(BaseModel):
    machineID: int
    model: str = Field(examples=["model3"])
    age: int
    telemetry: list[Reading] = Field(min_length=MIN_HISTORY, description="hourly readings, oldest first, no gaps")
    errors: list[ErrorEvent] = []
    maint: list[MaintEvent] = []


@lru_cache
def artifacts():
    mdir = model_dir()
    if not (mdir / "model.joblib").exists():
        raise HTTPException(503, "No trained model. Run `python train.py` first.")
    return joblib.load(mdir / "model.joblib"), json.loads((mdir / "meta.json").read_text())


@app.get("/health")
def health():
    _, meta = artifacts()
    return {"status": "ok", "model": meta["model"], "version": meta["version"]}


@app.post("/predict")
def predict(req: PredictRequest):
    model, meta = artifacts()
    mid = req.machineID
    tel = pd.DataFrame([r.model_dump() for r in req.telemetry]).assign(machineID=mid)
    gaps = tel["datetime"].sort_values().diff().dropna()
    if (gaps != pd.Timedelta(hours=1)).any():
        raise HTTPException(422, "telemetry must be hourly with no gaps or duplicates")
    errors = pd.DataFrame([e.model_dump() for e in req.errors], columns=["datetime", "errorID"]).assign(machineID=mid)
    maint = pd.DataFrame([m.model_dump() for m in req.maint], columns=["datetime", "comp"]).assign(machineID=mid)
    machines = pd.DataFrame([{"machineID": mid, "model": req.model, "age": req.age}])

    row = build_features(tel, errors, maint, machines).iloc[[-1]]
    risk = float(model.predict_proba(row[FEATURES])[0, 1])
    thr = meta["threshold"]
    return {
        "machineID": mid,
        "as_of": str(row["datetime"].iloc[0]),
        "horizon_hours": meta["horizon_hours"],
        "risk_score": round(risk, 4),
        "threshold": round(thr, 4),
        "alert": risk >= thr,
        "status": "ALERT" if risk >= thr else "WATCH" if risk >= thr / 2 else "OK",
        "errors_last_24h": int(row[[c for c in FEATURES if c.endswith("_count_24h")]].sum(axis=1).iloc[0]),
        "model_version": meta["version"],
    }
