"""Monitoring view: fleet risk scores, alerts for high-risk machines, per-machine history.

Run: streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from config import model_dir, report_dir

st.set_page_config(page_title="Predictive Maintenance", layout="wide")


@st.cache_data
def load():
    mdir = model_dir()
    meta = json.loads((mdir / "meta.json").read_text())
    scores = pd.read_csv(mdir / "test_scores.csv.gz", parse_dates=["datetime"])
    return meta, scores


if not (model_dir() / "meta.json").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

meta, scores = load()
thr, horizon = meta["threshold"], meta["horizon_hours"]
rows, events = meta["test"]["rows"], meta["test"]["events"]

st.title("Predictive Maintenance")
st.caption(f"Model: {meta['model']} | version {meta['version']} | predicts failure within {horizon}h | "
           f"alert threshold {thr:.2f}")
if meta["synthetic_data"]:
    st.warning("This model was trained on SYNTHETIC data. The numbers below are not real results.")

c = st.columns(5)
c[0].metric("PR-AUC", f"{rows['pr_auc']:.3f}", help=f"Random guessing would score {rows['prevalence']:.3f}")
c[1].metric("Recall", f"{rows['recall']:.0%}")
c[2].metric("Precision", f"{rows['precision']:.0%}")
c[3].metric("Failures caught", f"{events['failures_caught']} / {events['failures']}")
c[4].metric("False-alarm days / week", f"{events['false_alarm_days_per_week']:.1f}",
            help=f"Across {events['machines']} machines")

fleet_tab, machine_tab, model_tab = st.tabs(["Fleet monitor", "Machine history", "Model"])

with fleet_tab:
    times = sorted(scores["datetime"].unique())
    alert_times = sorted(scores.loc[scores["risk"] >= thr, "datetime"].unique())
    as_of = st.select_slider("Replay the test period - show the fleet as of", options=times,
                             value=alert_times[-1] if alert_times else times[-1],
                             format_func=lambda t: pd.Timestamp(t).strftime("%d %b %Y %H:%M"))
    now = scores[scores["datetime"] == as_of].sort_values("risk", ascending=False)
    now = now.assign(status=now["risk"].map(lambda r: "ALERT" if r >= thr else "WATCH" if r >= thr / 2 else "OK"))
    alerts = now[now["status"] == "ALERT"]
    if len(alerts):
        ids = ", ".join(f"#{m}" for m in alerts["machineID"])
        st.error(f"{len(alerts)} machine(s) at high risk of failing within {horizon}h: {ids}. Schedule inspection.")
    else:
        st.success("No machine is above the alert threshold at this time.")
    k = st.columns(3)
    k[0].metric("ALERT", int((now["status"] == "ALERT").sum()))
    k[1].metric("WATCH", int((now["status"] == "WATCH").sum()))
    k[2].metric("OK", int((now["status"] == "OK").sum()))
    cols = ["machineID", "status", "risk"] + [col for col in now.columns if col.endswith("_mean_24h")]
    st.dataframe(now[cols], hide_index=True, use_container_width=True, column_config={
        "risk": st.column_config.ProgressColumn("risk score", min_value=0.0, max_value=1.0, format="%.2f")})

with machine_tab:
    mid = st.selectbox("Machine", sorted(scores["machineID"].unique()))
    one = scores[scores["machineID"] == mid].set_index("datetime")
    st.subheader("Risk score over time")
    st.line_chart(one[["risk"]].assign(threshold=thr))
    pos = one[one["label"] == 1].reset_index()
    fails = sorted((pos["datetime"] + pd.to_timedelta(pos["hours_to_failure"], unit="h")).unique())
    if fails:
        st.write("Actual failures in this period: " + ", ".join(pd.Timestamp(f).strftime("%d %b %H:%M") for f in fails))
    else:
        st.write("No failures for this machine in this period.")
    st.subheader("Sensors (24h average)")
    sensor_cols = [col for col in one.columns if col.endswith("_mean_24h")]
    for col, box in zip(sensor_cols, st.columns(len(sensor_cols))):
        box.caption(col.replace("_mean_24h", ""))
        box.line_chart(one[col], height=180)

with model_tab:
    base_r, base_e = meta["baseline_any_error"]["rows"], meta["baseline_any_error"]["events"]
    st.subheader("Test period: model vs a simple rule")
    st.table(pd.DataFrame({
        "Model": [f"{rows['recall']:.1%}", f"{rows['precision']:.1%}", rows["false_alarms"],
                  f"{events['failures_caught']} of {events['failures']}",
                  f"{events['false_alarm_days_per_week']:.1f}"],
        "Rule: alert on any error in last 24h": [
            f"{base_r['recall']:.1%}", f"{base_r['precision']:.1%}", base_r["false_alarms"],
            f"{base_e['failures_caught']} of {base_e['failures']}", f"{base_e['false_alarm_days_per_week']:.1f}"],
    }, index=["Recall (rows)", "Precision (rows)", "False alarms (rows)", "Failures caught",
              "False-alarm days per week"]).astype(str))
    left, right = st.columns(2)
    with left:
        st.subheader("Candidates (validation PR-AUC)")
        st.table(pd.Series(meta["valid_pr_auc"], name="PR-AUC").round(4).astype(str))
        st.caption(f"Time split: train until {meta['split']['train_end'][:10]}, "
                   f"validation until {meta['split']['valid_end'][:10]}, test after.")
        if (report_dir() / "pr_curve.png").exists():
            st.image(str(report_dir() / "pr_curve.png"))
    with right:
        st.subheader("What drives the risk score")
        st.bar_chart(pd.DataFrame(meta["importance"], columns=["feature", "importance"]).set_index("feature"))
