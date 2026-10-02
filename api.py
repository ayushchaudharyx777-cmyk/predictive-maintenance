"""Scoring API. Send a machine's recent history, get its failure risk, the likely component and the reasons.

Run: uvicorn api:app --reload   -> docs at http://localhost:8000/docs
"""
import datetime as dt
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import ROOT, model_dir
from explain import reasons
from features import FEATURES, MIN_HISTORY, build_features

app = FastAPI(title="Predictive Maintenance API", version="2.0")


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
    comp_path = mdir / "component_model.joblib"
    return (joblib.load(mdir / "model.joblib"), json.loads((mdir / "meta.json").read_text()),
            joblib.load(comp_path) if comp_path.exists() else None)


def log_prediction(row: pd.DataFrame, risk: float, alert: bool) -> None:
    """Append the scored feature row to a JSONL file - this is what monitor.py checks for drift."""
    path = Path(os.environ.get("PDM_PRED_LOG", ROOT / "logs" / "predictions.jsonl"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"logged_at": dt.datetime.now(dt.timezone.utc).isoformat(), "risk": risk, "alert": alert,
                  "machineID": int(row["machineID"].iloc[0]), **row[FEATURES].iloc[0].to_dict()}
        with open(path, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")
    except OSError:
        pass  # a read-only disk must never take the API down


@app.get("/health")
def health():
    _, meta, _ = artifacts()
    return {"status": "ok", "model": meta["model"], "version": meta["version"]}


@app.post("/predict")
def predict(req: PredictRequest):
    model, meta, comp_model = artifacts()
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
    status = "ALERT" if risk >= thr else "WATCH" if risk >= thr / 2 else "OK"
    flagged = status != "OK"
    log_prediction(row, risk, risk >= thr)
    return {
        "machineID": mid,
        "as_of": str(row["datetime"].iloc[0]),
        "horizon_hours": meta["horizon_hours"],
        "risk_score": round(risk, 4),
        "threshold": round(thr, 4),
        "alert": risk >= thr,
        "status": status,
        "likely_component": str(comp_model.predict(row[FEATURES])[0]) if flagged and comp_model is not None else None,
        "reasons": reasons(model, row, meta["baseline"])[0] if flagged else [],
        "errors_last_24h": int(row[[c for c in FEATURES if c.endswith("_count_24h")]].sum(axis=1).iloc[0]),
        "model_version": meta["version"],
    }
