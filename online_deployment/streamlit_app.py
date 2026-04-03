import time
import datetime as dt
import pandas as pd
import streamlit as st

from prediction_service_loader import (
    list_entities_from_test_folder,
    load_test_array,
    load_ref_df,
    load_lstm,
    load_if,
)
from consistency_online import ks_consistency_same_as_offline
from anomaly_detection import anomaly_rate_lstm, anomaly_rate_if
from trust_score_generator import trust_score


st.set_page_config(page_title="Online Trust Generator", layout="wide")
st.title("Online Trust Generator (LSTM / IF)")

# -----------------------------
# Load ALL test data for ALL entities (cache in memory)
# -----------------------------
entities_all = sorted(list_entities_from_test_folder())

@st.cache_data(show_spinner=False)
def load_all_test_arrays(entities):
    data = {}
    for e in entities:
        data[e] = load_test_array(e)   # (T, 38)
    return data

all_test = load_all_test_arrays(entities_all)

# -----------------------------
# Entity selection (now it's instant, because already loaded)
# -----------------------------
entity = st.sidebar.selectbox("Entity", entities_all, index=0)

X_all = all_test[entity]
T = len(X_all)

st.sidebar.caption(f"Loaded ALL entities: {len(entities_all)}")
st.sidebar.caption(f"Current entity length: {T} rows (~{T} minutes)")


START_DT = dt.datetime(2025, 1, 1, 0, 0, 0)  # set start time：每分钟一条
END_DT = START_DT + dt.timedelta(minutes=max(T - 1, 0))

st.sidebar.caption(f"Loaded: {T} rows (≈ {T} minutes)")
st.sidebar.subheader("Select time range (slider)")

#  index slider，旁边显示时间
default_end_idx = min(500, max(T - 1, 1))
start_idx, end_idx = st.sidebar.slider(
    "Range (minutes index)",
    min_value=0,
    max_value=max(T - 1, 1),
    value=(0, default_end_idx),
    step=1
)

start_dt = START_DT + dt.timedelta(minutes=int(start_idx))
end_dt = START_DT + dt.timedelta(minutes=int(end_idx))

st.sidebar.caption(f"Start: {start_dt:%Y-%m-%d %H:%M}")
st.sidebar.caption(f"End:   {end_dt:%Y-%m-%d %H:%M}")

if end_idx <= start_idx:
    st.sidebar.error("End must be later than Start.")
    st.stop()

# Slice
X_seg = X_all[start_idx:end_idx]
segment_len = len(X_seg)

# -----------------------------
# Weights
# -----------------------------
st.sidebar.subheader("Trust weights")
w1 = st.sidebar.slider("w1 (1 - anomaly_rate)", 0.0, 1.0, 0.4, 0.05)
w2 = st.sidebar.slider("w2 (consistency)", 0.0, 1.0, 0.6, 0.05)

# -----------------------------
# Show selection context
# -----------------------------
st.subheader("Current Selection")
st.write({
    "entity": entity,
    "start_time": start_dt.strftime("%Y-%m-%d %H:%M"),
    "end_time": end_dt.strftime("%Y-%m-%d %H:%M"),
    "segment_len": int(segment_len),
})

# -----------------------------
# Consistency (always shown)
# -----------------------------
try:
    ref_df = load_ref_df(entity)
    cur_df = pd.DataFrame(X_seg)
    cons = ks_consistency_same_as_offline(ref_df, cur_df)
except Exception as e:
    st.error(f"Consistency failed: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("consistency (train vs selected)", f"{cons:.4f}")
c2.metric("segment_len", f"{segment_len}")
c3.metric("data_end", f"{END_DT:%Y-%m-%d %H:%M}")

# -----------------------------
# Buttons
# -----------------------------
colA, colB = st.columns(2)
run_lstm = colA.button("Run LSTM", use_container_width=True)
run_if = colB.button("Run IF", use_container_width=True)

# -----------------------------
# Helper for throughput
# -----------------------------
def windows_count(seg_len: int, w: int) -> int:
    return max(0, seg_len - w + 1)

# -----------------------------
# Results: LSTM
# -----------------------------
if run_lstm:
    try:
        lstm, scaler_lstm,window_size, thr_lstm = load_lstm(entity)
        n_win = windows_count(segment_len, window_size)

        t0 = time.perf_counter()
        ar = anomaly_rate_lstm(X_seg, lstm, scaler_lstm,window_size, thr_lstm)
        t1 = time.perf_counter()

        runtime = t1 - t0
        throughput = (n_win / runtime) if runtime > 0 else 0.0
        tscore = trust_score(ar, cons, w1=w1, w2=w2)

        st.success("LSTM Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("anomaly_rate (LSTM)", f"{ar:.4f}")
        r2.metric("runtime (sec)", f"{runtime:.4f}")
        r3.metric("throughput (windows/s)", f"{throughput:,.1f}")
        r4.metric("trust (LSTM)", f"{tscore:.4f}")

        st.caption(f"LSTM: window_size={window_size}, threshold={thr_lstm:.6f}, windows={n_win}")

    except Exception as e:
        st.error(f"LSTM failed: {e}")

# -----------------------------
# Results: IF
# -----------------------------
if run_if:
    try:
        ifm, scaler_if, thr_if = load_if(entity)
        _, _, window_size, _ = load_lstm(entity)
        n_win = windows_count(segment_len, window_size)

        t0 = time.perf_counter()
        ar = anomaly_rate_if(X_seg, ifm, scaler_if,thr_if)  
        t1 = time.perf_counter()

        runtime = t1 - t0
        throughput = (n_win / runtime) if runtime > 0 else 0.0
        tscore = trust_score(ar, cons, w1=w1, w2=w2)

        st.success("IF Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("anomaly_rate (IF)", f"{ar:.4f}")
        r2.metric("runtime (sec)", f"{runtime:.4f}")
        r3.metric("throughput (windows/s)", f"{throughput:,.1f}")
        r4.metric("trust (IF)", f"{tscore:.4f}")

        # IF 真实期望维度（调试/论文解释很好用）
        expected = getattr(ifm, "n_features_in_", None)
        st.caption(f"IF: threshold={thr_if:.6f}, model_n_features_in_={expected}")

    except Exception as e:
        st.error(f"IF failed: {e}")
