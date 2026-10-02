import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(dirs):
    import api

    api.artifacts.cache_clear()
    with TestClient(api.app) as c:
        yield c


def _payload(raw, hours=72, before_failure=True):
    fail = raw["failures"].iloc[0]
    mid = int(fail["machineID"])
    end = fail["datetime"] - pd.Timedelta(hours=2) if before_failure else fail["datetime"] - pd.Timedelta(days=4)
    tel = raw["telemetry"]
    tel = tel[(tel["machineID"] == mid) & (tel["datetime"] <= end)].tail(hours)
    machine = raw["machines"].set_index("machineID").loc[mid]

    def events(df, col):
        d = df[(df["machineID"] == mid) & (df["datetime"] <= end)]
        return [{"datetime": str(t), col: v} for t, v in zip(d["datetime"], d[col])]

    return {
        "machineID": mid, "model": machine["model"], "age": int(machine["age"]),
        "telemetry": [{"datetime": str(r.datetime), "volt": r.volt, "rotate": r.rotate, "pressure": r.pressure,
                       "vibration": r.vibration} for r in tel.itertuples()],
        "errors": events(raw["errors"], "errorID"), "maint": events(raw["maint"], "comp"),
    }


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_flags_machine_about_to_fail(client, raw):
    risky = client.post("/predict", json=_payload(raw)).json()
    calm = client.post("/predict", json=_payload(raw, before_failure=False)).json()
    assert 0 <= calm["risk_score"] <= 1
    assert risky["risk_score"] > calm["risk_score"]
    assert risky["status"] in {"ALERT", "WATCH", "OK"} and risky["horizon_hours"] == 24


def test_predict_works_without_error_or_maintenance_history(client, raw):
    body = dict(_payload(raw), errors=[], maint=[])
    assert client.post("/predict", json=body).status_code == 200


def test_too_little_history_rejected(client, raw):
    assert client.post("/predict", json=_payload(raw, hours=10)).status_code == 422


def test_gap_rejected(client, raw):
    body = _payload(raw)
    del body["telemetry"][20]
    assert client.post("/predict", json=body).status_code == 422
