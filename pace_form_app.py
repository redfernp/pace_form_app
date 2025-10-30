# pace_form_app.py — Single-file Streamlit app
# A repeatable pace & class analysis workflow with realistic pace rules:
# - Front-aware early-energy model
# - No-front cap
# - Dominant-front cap (new)
# - Single-credible-front cap
# - Weak solo leader downgrade
# Usage: streamlit run pace_form_app.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import io

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Data Models & Settings
# -----------------------------
@dataclass
class HorseRow:
    horse: str
    run_styles: List[int]  # last five, values 0..4; 0 ignored
    adj_speed: Optional[float]  # Adjusted Speed Figure (may be None)
    or_today: Optional[int] = None
    or_highest_win: Optional[int] = None

@dataclass
class Settings:
    class_par: float = 77.0
    distance_f: float = 7.0
    # Run style thresholds
    front_thr: float = 1.6
    prom_thr: float = 2.4
    mid_thr: float = 3.0
    # Leader Credibility thresholds (Δ vs Par)
    lcp_high: float = -3.0
    lcp_question_low: float = -8.0  # −8 to −4 questionable
    # Suitability weights
    wp_even: float = 0.5
    wp_confident: float = 0.65  # when pace is predictable (slow or very strong)

STYLE_LABELS = {1: "Front", 2: "Prominent", 3: "Mid", 4: "Hold-up"}
PACE_ORDER = ["Slow", "Even", "Strong", "Very Strong"]

PACEFIT = {
    "Slow": {"Front": 5, "Prominent": 4, "Mid": 3, "Hold-up": 2},
    "Even": {"Front": 4, "Prominent": 5, "Mid": 4, "Hold-up": 3},
    "Strong": {"Front": 2, "Prominent": 3, "Mid": 4, "Hold-up": 5},
    "Very Strong": {"Front": 1, "Prominent": 2, "Mid": 4, "Hold-up": 5},
}

# -----------------------------
# Core Helpers
# -----------------------------

def avg_style(run_styles: List[int]) -> float:
    vals = [int(x) for x in run_styles if int(x) != 0]
    return float(np.mean(vals)) if vals else np.nan


def classify_style(avg: float, s: Settings) -> str:
    if np.isnan(avg):
        return "Unknown"
    if avg < s.front_thr:
        return "Front"
    if avg < s.prom_thr:
        return "Prominent"
    if avg < s.mid_thr:
        return "Mid"
    return "Hold-up"


def delta_vs_par(adj_speed: Optional[float], class_par: float) -> Optional[float]:
    return None if adj_speed is None else round(float(adj_speed) - class_par, 1)


def lcp_from_delta(style: str, dvp: Optional[float], s: Settings) -> str:
    if style not in ("Front", "Prominent") or dvp is None:
        return "N/A"
    if dvp >= s.lcp_high:
        return "High"
    if dvp >= s.lcp_question_low:
        return "Questionable"
    return "Unlikely"


def speed_score(dvp: Optional[float]) -> float:
    if dvp is None:
        return 3.0
    if dvp >= 2:
        return 5.0
    if dvp >= -1:
        return 4.0
    if dvp >= -4:
        return 3.0
    if dvp >= -8:
        return 2.0
    return 1.0

# -----------------------------
# Pace Projection (front-aware + caps + weak solo leader + dominant front)
# -----------------------------

def project_pace(rows: List[HorseRow], s: Settings) -> Tuple[str, float, Dict[str, str]]:
    """
    Front-aware pace model with realistic caps.

    Rules encoded:
    1) Early-energy index (Front > Prominent; Questionables discounted).
    2) NO-FRONT CAP: If no Fronts (or no *High* Fronts), cap at 'Even' unless there are
       ≥3 High Prominent and their mean ΔvsPar ≥ -1 (near par).
    3) DOMINANT-FRONT CAP (NEW):
       If exactly one High Front with ΔvsPar ≥ +2 and ≤1 High Prominent,
       cap Strong/Very Strong to 'Even' (leader likely to dictate).
    4) SINGLE-FRONT CAP (modest edge):
       If exactly one High Front and its ΔvsPar ≤ +1, cap to 'Even'.
       If that lone Front has ΔvsPar ≤ -2 and there are ≤1 High Prominent → allow 'Slow'.
    5) WEAK SOLO LEADER:
       If exactly one Front in the whole field and its ΔvsPar ≤ -8, downgrade one category.
    """
    if not rows:
        return "N/A", 0.0, {}

    # Build frame
    recs = []
    for r in rows:
        avg = avg_style(r.run_styles)
        style = classify_style(avg, s)
        dvp = delta_vs_par(r.adj_speed, s.class_par)
        lcp  = lcp_from_delta(style, dvp, s)
        recs.append(dict(horse=r.horse, style=style, dvp=dvp, lcp=lcp))
    d = pd.DataFrame(recs)
    if d.empty:
        return "N/A", 0.0, {}

    # Credibility slices
    front_all   = d[d["style"] == "Front"]
    prom_all    = d[d["style"] == "Prominent"]

    front_high  = d[(d["style"] == "Front") & (d["lcp"] == "High")]
    prom_high   = d[(d["style"] == "Prominent") & (d["lcp"] == "High")]
    front_q     = d[(d["style"] == "Front") & (d["lcp"] == "Questionable")]
    prom_q      = d[(d["style"] == "Prominent") & (d["lcp"] == "Questionable")]

    n_front        = len(front_all)
    n_front_high   = len(front_high)
    n_prom_high    = len(prom_high)
    n_front_q      = len(front_q)
    n_prom_q       = len(prom_q)

    # 1) Early energy (tunable weights)
    W_FH, W_PH, W_FQ, W_PQ = 2.0, 0.8, 0.5, 0.2
    early_energy = (W_FH*n_front_high) + (W_PH*n_prom_high) + (W_FQ*n_front_q) + (W_PQ*n_prom_q)

    # Base scenario from energy / credible count
    if n_front_high >= 2:
        scenario, conf = "Strong", 0.65
    elif (n_front_high + n_prom_high) >= 3 and early_energy >= 3.2:
        scenario, conf = "Strong", 0.65
    elif early_energy >= 3.2:
        scenario, conf = "Strong", 0.60
    elif (n_front_high + n_prom_high) >= 2:
        scenario, conf = "Even", 0.60
    elif (n_front_high + n_prom_high) == 1 and (n_front_q + n_prom_q) >= 1:
        scenario, conf = "Even", 0.55
    elif (n_front_high + n_prom_high) == 1:
        scenario, conf = "Even", 0.60
    elif (n_front_q + n_prom_q) >= 1:
        scenario, conf = "Slow", 0.60
    else:
        scenario, conf = "Slow", 0.70

    # 2) NO-FRONT CAP (no *effective* Fronts)
    if n_front == 0 or n_front_high == 0:
        allow_strong = False
        if n_prom_high >= 3:
            try:
                allow_strong = float(prom_high["dvp"].mean()) >= -1.0  # near par
            except Exception:
                allow_strong = False
        if not allow_strong and scenario in ("Strong", "Very Strong"):
            scenario, conf = "Even", min(conf, 0.60)

    # 3) DOMINANT-FRONT CAP (NEW)
    if n_front_high == 1 and n_prom_high <= 1:
        try:
            lone_front_dvp = float(front_high["dvp"].iloc[0])
        except Exception:
            lone_front_dvp = None
        if (lone_front_dvp is not None) and (lone_front_dvp >= 2.0):
            if scenario in ("Strong", "Very Strong"):
                scenario, conf = "Even", max(conf, 0.65)

    # 4) SINGLE-FRONT CAP (modest edge or below par)
    if n_front_high == 1:
        try:
            lone_front_dvp = float(front_high["dvp"].iloc[0])
        except Exception:
            lone_front_dvp = None

        if (lone_front_dvp is None) or (lone_front_dvp <= 1.0):
            if scenario in ("Strong", "Very Strong"):
                scenario, conf = "Even", max(conf, 0.60)

        if (lone_front_dvp is not None) and (lone_front_dvp <= -2.0) and (n_prom_high <= 1):
            if scenario == "Even":
                scenario, conf = "Slow", max(conf, 0.65)

    # 5) WEAK SOLO LEADER (any single Front present and well below par)
    if n_front == 1:
        try:
            dvp_front_any = float(front_all["dvp"].iloc[0])
        except Exception:
            dvp_front_any = None
        if (dvp_front_any is not None) and (dvp_front_any <= -8.0):
            idx = max(0, PACE_ORDER.index(scenario) - 1)
            scenario = PACE_ORDER[idx]
            conf = max(conf, 0.65)

    lcp_map = dict(zip(d["horse"], d["lcp"]))
    return scenario, conf, lcp_map

# -----------------------------
# Suitability Scoring
# -----------------------------

def suitability(rows: List[HorseRow], s: Settings) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    scenario, conf, lcp_map = project_pace(rows, s)
    if scenario == "N/A":
        return pd.DataFrame()

    wp = s.wp_confident if scenario in ("Slow", "Very Strong") and conf >= 0.65 else s.wp_even
    ws = 1 - wp

    out = []
    for r in rows:
        avg = avg_style(r.run_styles)
        style = classify_style(avg, s)
        dvp = delta_vs_par(r.adj_speed, s.class_par)
        pacefit = PACEFIT[scenario].get(style, 3)
        speedfit = speed_score(dvp)
        score = round(pacefit * wp + speedfit * ws, 1)
        out.append({
            "Horse": r.horse,
            "AvgStyle": round(avg, 2),
            "Style": style,
            "ΔvsPar": dvp,
            "LCP": lcp_map.get(r.horse, "N/A"),
            "PaceFit": pacefit,
            "SpeedFit": speedfit,
            "wp": wp,
            "ws": ws,
            "Suitability": score,
            "Scenario": scenario,
            "Confidence": conf,
        })
    df = pd.DataFrame(out).sort_values(["Suitability", "SpeedFit"], ascending=False)
    return df

# -----------------------------
# Input Normalization (two-file mode)
# -----------------------------

def normalize_two_files(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    REQUIRED = [
        "Horse","RS_Lto1","RS_Lto2","RS_Lto3","RS_Lto4","RS_Lto5",
        "AdjSpeed","OR","OR_HighestWin"
    ]
    aliases = {
        "horse": "Horse", "horse name": "Horse", "horse_name": "Horse", "name": "Horse", "runner": "Horse",
        "adj_speed": "AdjSpeed", "speed": "AdjSpeed", "key_speed": "AdjSpeed", "key speed factors average": "AdjSpeed",
        "or": "OR", "or_today": "OR", "or_today_lb": "OR", "or today": "OR",
        "or_high": "OR_HighestWin", "highest_win_or": "OR_HighestWin", "highest winning or": "OR_HighestWin",
        "lto1": "RS_Lto1","lto2":"RS_Lto2","lto3":"RS_Lto3","lto4":"RS_Lto4","lto5":"RS_Lto5",
        "rs1":"RS_Lto1","rs2":"RS_Lto2","rs3":"RS_Lto3","rs4":"RS_Lto4","rs5":"RS_Lto5",
    }

    def std(df):
        cols = {c: aliases.get(str(c).strip().lower(), c) for c in df.columns}
        out = df.rename(columns=cols).copy()
        if "Horse" in out.columns:
            out["Horse"] = (
                out["Horse"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            )
        return out

    A = std(df_a)
    B = std(df_b)

    if "Horse" not in A.columns:
        raise ValueError(f"File A missing 'Horse' column after normalization. Found: {list(A.columns)}")
    if "Horse" not in B.columns:
        raise ValueError(f"File B missing 'Horse' column after normalization. Found: {list(B.columns)}")

    keep_a = [c for c in A.columns if c in ["Horse","RS_Lto1","RS_Lto2","RS_Lto3","RS_Lto4","RS_Lto5"]]
    keep_b = [c for c in B.columns if c in ["Horse","AdjSpeed","OR","OR_HighestWin"]]

    A = A[keep_a].copy()
    B = B[keep_b].copy()

    merged = pd.merge(A, B, on="Horse", how="left")

    for col in REQUIRED:
        if col not in merged.columns:
            merged[col] = None

    for c in [f"RS_Lto{i}" for i in range(1,6)]:
        merged[c] = merged[c].fillna(0).astype(int)

    info = f"Normalized columns: {list(merged.columns)}. Rows: {len(merged)}"
    return merged[REQUIRED], info

# -----------------------------
# Utility: Build HorseRows from a merged dataframe
# -----------------------------

def to_rows(df: pd.DataFrame) -> List[HorseRow]:
    rows: List[HorseRow] = []
    for _, r in df.iterrows():
        styles = [int(r.get(f"RS_Lto{i}", 0)) for i in range(1, 6)]
        adj = None if pd.isna(r.get("AdjSpeed")) else float(r["AdjSpeed"])
        rows.append(HorseRow(
            horse=str(r["Horse"]),
            run_styles=styles,
            adj_speed=adj,
            or_today=None if pd.isna(r.get("OR")) else int(r["OR"]),
            or_highest_win=None if pd.isna(r.get("OR_HighestWin")) else int(r["OR_HighestWin"]),
        ))
    return rows

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="PaceForm – One-File App", page_icon="🏇", layout="wide")
st.title("🏇 PaceForm – Pace & Class Analysis (One-file App)")

st.sidebar.header("Race Settings")
s = Settings()
s.class_par = st.sidebar.number_input("Class Par", value=float(s.class_par))
s.distance_f = st.sidebar.number_input("Distance (f)", value=float(s.distance_f))

st.sidebar.header("Thresholds")
s.front_thr = st.sidebar.number_input("Front threshold (<)", value=float(s.front_thr), step=0.1)
s.prom_thr = st.sidebar.number_input("Prominent threshold (<)", value=float(s.prom_thr), step=0.1)
s.mid_thr = st.sidebar.number_input("Mid threshold (<)", value=float(s.mid_thr), step=0.1)

st.sidebar.header("Leader Credibility (Δ vs Par)")
s.lcp_high = st.sidebar.number_input("High ≥", value=float(s.lcp_high), step=0.5)
s.lcp_question_low = st.sidebar.number_input("Questionable lower bound", value=float(s.lcp_question_low), step=0.5)

st.sidebar.header("Weights")
s.wp_even = st.sidebar.slider("wp (Even/Uncertain)", 0.3, 0.8, float(s.wp_even), 0.05)
s.wp_confident = st.sidebar.slider("wp (Predictable Slow/Very Strong)", 0.3, 0.8, float(s.wp_confident), 0.05)

st.markdown("#### Upload input files (both files required)")
left, right = st.columns(2)
with left:
    f1 = st.file_uploader("File A (horses & run styles)", type=["csv"], key="file_a")
with right:
    f2 = st.file_uploader("File B (speed & ratings)", type=["csv"], key="file_b")

if not (f1 and f2):
    st.info("Please upload both CSV files to generate the analysis.")
    st.stop()

try:
    df, info = normalize_two_files(pd.read_csv(f1), pd.read_csv(f2))
    st.success("Two files normalized ✔")
    st.caption(info)
except Exception as e:
    st.error(f"Failed to read/normalize files: {e}")
    st.stop()

st.dataframe(df, use_container_width=True, hide_index=True)

rows = to_rows(df)
res = suitability(rows, s)
if res.empty:
    st.warning("No analysable rows after normalization. Check your input files.")
    st.stop()

scenario = res.iloc[0]["Scenario"]
conf = float(res.iloc[0]["Confidence"]) if "Confidence" in res.columns else 0.0

st.subheader(f"Projected Pace: {scenario} (confidence {conf:.2f})")

cols = st.columns([1.6, 1.4])
with cols[0]:
    st.markdown("### Suitability Ratings")
    st.dataframe(
        res[["Horse","Style","ΔvsPar","LCP","PaceFit","SpeedFit","wp","ws","Suitability"]],
        use_container_width=True,
    )
with cols[1]:
    st.markdown("### Run Style Summary")
    show = res[["Horse","AvgStyle","Style","LCP","ΔvsPar"]].copy()
    st.dataframe(show, use_container_width=True)

st.markdown("### Final Verdict")
short = res.sort_values(["Suitability","SpeedFit"], ascending=False).head(3)[[
    "Horse","Suitability","Style","ΔvsPar"
]]
for i, row in short.reset_index(drop=True).iterrows():
    medals = ["🥇","🥈","🥉"][i]
    st.write(f"{medals} **{row['Horse']}** – Score {row['Suitability']} | {row['Style']} | ΔvsPar {row['ΔvsPar']}")

csv_bytes = res.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv_bytes,
    file_name="paceform_results.csv",
    mime="text/csv",
)

st.caption("Front-aware model with no-front cap, dominant-front cap, single-front cap, and weak solo leader adjustments.")
