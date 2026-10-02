"""Monitoring view: fleet risk scores, alerts with reasons, alert log, savings, drift.

Run: streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from config import model_dir, report_dir
from evaluate import alert_episodes

st.set_page_config(page_title="Predictive Maintenance", layout="wide")


@st.cache_data
def load(version_key: float):
    mdir = model_dir()
    meta = json.loads((mdir / "meta.json").read_text())
    scores = pd.read_csv(mdir / "test_scores.csv.gz", parse_dates=["datetime"], keep_default_na=False,
                         na_values={"hours_to_failure": [""]})
    return meta, scores


def table(df: pd.DataFrame, **kwargs) -> None:
    """Full-width table that works on both old and new Streamlit versions."""
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    except Exception:  # noqa: BLE001 - older Streamlit: width must be a number
        st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


def status_of(risk: float, thr: float) -> str:
    return "ALERT" if risk >= thr else "WATCH" if risk >= thr / 2 else "OK"


if not (model_dir() / "meta.json").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

meta, scores = load((model_dir() / "meta.json").stat().st_mtime)
thr, horizon = meta["threshold"], meta["horizon_hours"]
rows, events = meta["test"]["rows"], meta["test"]["events"]

st.title("Predictive Maintenance")
st.caption(f"Model: {meta['model']} | version {meta['version']} | predicts failure within {horizon}h | "
           f"alert threshold {thr:.2f}")
if meta["synthetic_data"]:
    st.warning("This model was trained on SYNTHETIC data. The numbers below are not real results.")

c = st.columns(5)
c[0].metric("PR-AUC", f"{rows['pr_auc']:.3f}",
            help=f"95% CI {meta['pr_auc_ci'][0]:.3f}-{meta['pr_auc_ci'][1]:.3f}. Random guessing: {rows['prevalence']:.3f}")
c[1].metric("Recall", f"{rows['recall']:.0%}")
c[2].metric("Precision", f"{rows['precision']:.0%}")
c[3].metric("Failures caught", f"{events['failures_caught']} / {events['failures']}",
            help=f"Median warning {events['median_lead_hours']:.0f}h before the failure")
c[4].metric("False-alarm days / week", f"{events['false_alarm_days_per_week']:.1f}",
            help=f"Across {events['machines']} machines")

fleet_tab, log_tab, machine_tab, model_tab, drift_tab = st.tabs(
    ["Fleet monitor", "Alert log", "Machine history", "Model and savings", "Drift"])

with fleet_tab:
    times = sorted(scores["datetime"].unique())
    alert_times = sorted(scores.loc[scores["risk"] >= thr, "datetime"].unique())
    as_of = st.select_slider("Replay the test period - show the fleet as of", options=times,
                             value=alert_times[-1] if alert_times else times[-1],
                             format_func=lambda t: pd.Timestamp(t).strftime("%d %b %Y %H:%M"))
    now = scores[scores["datetime"] == as_of].sort_values("risk", ascending=False)
    now = now.assign(status=now["risk"].map(lambda r: status_of(r, thr)))
    alerts = now[now["status"] == "ALERT"]
    if len(alerts):
        st.error(f"{len(alerts)} machine(s) at high risk of failing within {horizon}h. Schedule inspection:")
        for a in alerts.itertuples():
            part = f" - likely **{a.component}**" if a.component else ""
            why = f" - {a.reasons}" if a.reasons else ""
            st.markdown(f"- **Machine #{a.machineID}** (risk {a.risk:.2f}){part}{why}")
    else:
        st.success("No machine is above the alert threshold at this time.")
    k = st.columns(3)
    k[0].metric("ALERT", int((now["status"] == "ALERT").sum()))
    k[1].metric("WATCH", int((now["status"] == "WATCH").sum()))
    k[2].metric("OK", int((now["status"] == "OK").sum()))
    cols = ["machineID", "status", "risk", "component", "reasons"] + [x for x in now.columns if x.endswith("_mean_24h")]
    table(now[cols], column_config={
        "risk": st.column_config.ProgressColumn("risk score", min_value=0.0, max_value=1.0, format="%.2f"),
        "component": "likely component", "reasons": "why"})

with log_tab:
    episodes = alert_episodes(scores, thr)
    followed = int((episodes["outcome"] == "Failure followed").sum())
    k = st.columns(3)
    k[0].metric("Alert episodes", len(episodes))
    k[1].metric("Followed by a failure", followed)
    k[2].metric("False alarms", len(episodes) - followed)
    st.caption("Consecutive alerts on one machine are grouped into one episode. Lead = hours from the first alert "
               "to the failure.")
    outcome = st.radio("Show", ["All", "Failure followed", "False alarm"], horizontal=True)
    shown = episodes if outcome == "All" else episodes[episodes["outcome"] == outcome]
    table(shown, column_config={
        "peak_risk": st.column_config.NumberColumn("peak risk", format="%.2f"),
        "lead_hours": st.column_config.NumberColumn("lead (h)", format="%.0f"),
        "component": "likely component", "reasons": "why"})
    st.download_button("Download alert log (CSV)", shown.to_csv(index=False), "alert_log.csv", "text/csv")

with machine_tab:
    mid = st.selectbox("Machine", sorted(scores["machineID"].unique()))
    one = scores[scores["machineID"] == mid].set_index("datetime")
    st.subheader("Risk score over time")
    st.line_chart(one[["risk"]].assign(threshold=thr))
    pos = one[one["label"] == 1].reset_index()
    fails = sorted((pos["datetime"] + pd.to_timedelta(pos["hours_to_failure"].astype(float), unit="h")).unique())
    if fails:
        st.write("Actual failures in this period: " + ", ".join(pd.Timestamp(f).strftime("%d %b %H:%M") for f in fails))
    else:
        st.write("No failures for this machine in this period.")
    st.subheader("Sensors (24h average)")
    sensor_cols = [x for x in one.columns if x.endswith("_mean_24h")]
    for col, box in zip(sensor_cols, st.columns(len(sensor_cols))):
        box.caption(col.replace("_mean_24h", ""))
        box.line_chart(one[col], height=180)

with model_tab:
    k = meta["costs"]
    st.subheader("What each policy would have cost over the test period")
    st.caption(f"ASSUMED costs: breakdown {k['unplanned_failure']:,}, planned repair {k['planned_repair']:,}, "
               f"inspection after a false alarm {k['inspection']:,}. Change them in params.yaml.")
    pol = pd.DataFrame(meta["policy"]).rename(columns={
        "policy": "Policy", "failures_caught": "Caught", "failures_missed": "Missed",
        "false_alarm_days": "False-alarm days", "cost": "Cost",
        "saving_vs_run_to_failure": "Saving vs run-to-failure", "saving_pct": "Saving %"})
    table(pol, column_config={
        "Cost": st.column_config.NumberColumn(format="%d"),
        "Saving vs run-to-failure": st.column_config.NumberColumn(format="%d"),
        "Saving %": st.column_config.NumberColumn(format="%.0f%%")})
    comp = meta["component"]
    if comp:
        st.write(f"**Which component will fail:** correct for {comp['accuracy']:.1%} of {comp['n']} warning rows "
                 f"(rule 'most overdue component': {comp['baseline_accuracy']:.1%}).")
    left, right = st.columns(2)
    with left:
        st.subheader("Candidates (validation PR-AUC)")
        st.table(pd.Series(meta["valid_pr_auc"], name="PR-AUC").round(4).astype(str))
        st.caption(f"Time split: train until {meta['split']['train_end'][:10]}, "
                   f"validation until {meta['split']['valid_end'][:10]}, test after. "
                   f"Threshold chosen on validation by {meta['threshold_by']}.")
        if (report_dir() / "pr_curve.png").exists():
            st.image(str(report_dir() / "pr_curve.png"))
    with right:
        st.subheader("What drives the risk score")
        st.bar_chart(pd.DataFrame(meta["importance"], columns=["feature", "importance"]).set_index("feature"))

with drift_tab:
    path = report_dir() / "drift_report.json"
    if not path.exists():
        st.info("No drift report yet. Run `python monitor.py --simulate` for a demo, or "
                "`python monitor.py --current logs/predictions.jsonl` to check real API traffic.")
    else:
        rep = json.loads(path.read_text())
        show = {"ALERT": st.error, "WARN": st.warning, "OK": st.success}[rep["status"]]
        show(f"Drift status: {rep['status']} - {rep['rows_checked']} rows checked on {rep['generated']}")
        st.write(f"Source: {rep.get('source', 'unknown')}")
        st.caption("PSI compares each feature's distribution with the training data: "
                   "< 0.10 OK, 0.10-0.25 WARN, > 0.25 ALERT.")
        table(pd.DataFrame(rep["features"]))
