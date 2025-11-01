# pace_form_app.py — Single-file Streamlit app
# A repeatable pace & class analysis workflow with realistic pace rules:
# - Front-aware early-energy model
# - No-front cap
# - Dominant-front cap
# - Single-credible-front cap
# - Weak solo leader downgrade
# - Distance-aware sprint handling for 5f & 6f:
#     * Sprint cap (Strong → Even) when a lone credible front is near/above par with limited pressers
#     * Sprint-specific PaceFit maps (position > power at short trips)
#     * Higher pace-weighting at 5–6f
# ...plus a "Reason used" debug section explaining the choice
# Usage: streamlit run pace_form_app.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

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

# Default PaceFit (routes / generic)
PACEFIT = {
    "Slow":   {"Front": 5, "Prominent": 4, "Mid": 3, "Hold-up": 2},
    "Even":   {"Front": 4, "Prominent": 5, "Mid": 4, "Hold-up": 3},
    "Strong": {"Front": 2, "Prominent": 3, "Mid": 4, "Hold-up": 5},
    "Very Strong": {"Front": 1, "Prominent": 2, "Mid": 4, "Hold-up": 5},
}

# Sprint-specific PaceFit maps
PACEFIT_5F = {
    "Slow":   {"Front": 5.0, "Prominent": 4.5, "Mid": 3.5, "Hold-up": 2.0},
    "Even":   {"Front": 4.5, "Prominent": 5.0, "Mid": 3.5, "Hold-up": 2.5},
    "Strong": {"Front": 3.5, "Prominent": 4.5, "Mid": 4.0, "Hold-up": 3.0},
}
PACEFIT_6F = {
    "Slow":   {"Front": 5.0, "Prominent": 4.5, "Mid": 3.5, "Hold-up": 2.5},
    "Even":   {"Front": 4.0, "Prominent": 5.0, "Mid": 4.0, "Hold-up": 3.0},
    "Strong": {"Front": 3.0, "Prominent": 4.0, "Mid": 4.5, "Hold-up": 3.5},
}

# -----------------------------
# Core Helpers
# -----------------------------

def _dist_band(d: float) -> str:
    if d <= 5.5:
        return "5f"
    if d <= 6.5:
        return "6f"
    return "route"


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
        return 2.3
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
# Pace Projection (front-aware + caps)
# -----------------------------

def project_pace(rows: List[HorseRow], s: Settings) -> Tuple[str, float, Dict[str, str], Dict[str, object]]:
    debug = {"rules_applied": []}
    if not rows:
        return "N/A", 0.0, {}, {"error": "no rows", "rules_applied": []}

    recs = []
    for r in rows:
        avg = avg_style(r.run_styles)
        style = classify_style(avg, s)
        dvp = delta_vs_par(r.adj_speed, s.class_par)
        lcp  = lcp_from_delta(style, dvp, s)
        recs.append(dict(horse=r.horse, style=style, dvp=dvp, lcp=lcp))
    d = pd.DataFrame(recs)
    if d.empty:
        return "N/A", 0.0, {}, {"error": "empty dataframe", "rules_applied": []}

    front_high = d[(d["style"] == "Front") & (d["lcp"] == "High")]
    prom_high = d[(d["style"] == "Prominent") & (d["lcp"] == "High")]
    front_q = d[(d["style"] == "Front") & (d["lcp"] == "Questionable")]
    prom_q = d[(d["style"] == "Prominent") & (d["lcp"] == "Questionable")]

    n_front_high, n_prom_high = len(front_high), len(prom_high)
    n_front_q, n_prom_q = len(front_q), len(prom_q)

    W_FH, W_PH, W_FQ, W_PQ = 2.0, 0.8, 0.5, 0.2
    early_energy = (W_FH*n_front_high) + (W_PH*n_prom_high) + (W_FQ*n_front_q) + (W_PQ*n_prom_q)

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

    debug.update({"early_energy": early_energy})
    lcp_map = dict(zip(d["horse"], d["lcp"]))
    return scenario, conf, lcp_map, debug

# -----------------------------
# Suitability Scoring
# -----------------------------

def suitability(rows: List[HorseRow], s: Settings) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not rows:
        return pd.DataFrame(), {"error": "no rows"}
    scenario, conf, lcp_map, debug = project_pace(rows, s)
    if scenario == "N/A":
        return pd.DataFrame(), {"error": "pace N/A"}

    band = _dist_band(getattr(s, "distance_f", 7.0))
    if band == "5f":
        pacefit_map = PACEFIT_5F[scenario]
    elif band == "6f":
        pacefit_map = PACEFIT_6F[scenario]
    else:
        pacefit_map = PACEFIT[scenario]

    wp = s.wp_confident if scenario in ("Slow", "Very Strong") and conf >= 0.65 else s.wp_even
    if band == "5f":
        wp = max(wp, 0.60)
    elif band == "6f":
        wp = max(wp, 0.55)
    ws = 1 - wp

    missing_count = sum(1 for r in rows if r.adj_speed is None)
    missing_ratio = missing_count / max(len(rows), 1)
    if missing_ratio > 0.5:
        ws *= 0.8
        wp = 1 - ws
    debug["missing_speed_ratio"] = float(missing_ratio)

    out = []
    for r in rows:
        avg = avg_style(r.run_styles)
        style = classify_style(avg, s)
        dvp = delta_vs_par(r.adj_speed, s.class_par)
        pacefit = pacefit_map.get(style, 3)
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
    return df, debug

# -----------------------------
# Trading Suggestions (pace-driven)
# -----------------------------

def trading_suggestions(res: pd.DataFrame, debug: Dict[str, object]) -> pd.DataFrame:
    if res.empty:
        return pd.DataFrame()
    scenario = str(res.iloc[0].get("Scenario", "Even"))
    conf = float(res.iloc[0].get("Confidence", 0.6))
    band = str(debug.get("distance_band", "route"))

    df = res.copy()
    # Helpers
    is_front_h = (df["Style"] == "Front") & (df["LCP"] == "High")
    is_front_q = (df["Style"] == "Front") & (df["LCP"].isin(["Questionable", "Unlikely"]))
    is_prom    = (df["Style"] == "Prominent")
    is_mid     = (df["Style"] == "Mid")
    is_hold    = (df["Style"] == "Hold-up")

    plays: List[Dict[str, object]] = []

    def add_rows(mask, play, entry_note, exit_note, rationale, limit=3):
        sub = df[mask].sort_values(["Suitability", "ΔvsPar"], ascending=False).head(limit)
        for _, r in sub.iterrows():
            plays.append({
                "Horse": r["Horse"],
                "Style": r["Style"],
                "LCP": r["LCP"],
                "ΔvsPar": r["ΔvsPar"],
                "Suitability": r["Suitability"],
                "Play": play,
                "Entry": entry_note,
                "Exit": exit_note,
                "Why": rationale,
            })

    # Scenario-specific logic
    if scenario == "Slow":
        add_rows(is_prom, "Back-to-Lay",
                 "Back pre-off (handy stalk)",
                 "Lay @ ~0.6–0.7× entry IR",
                 "Prominents travel best in Slow races; efficient fractions compress price.")
        add_rows(is_front_q, "Lay-to-Back",
                 "Small lay pre-off",
                 "Back @ 2–3× entry IR",
                 "Weak/unchallenged fronts tend to drift when challenged late.", limit=2)

    elif scenario == "Even":
        add_rows(is_front_h, "Back-to-Lay",
                 "Back pre-off (credible leader)",
                 "Lay @ ~0.5–0.7× entry IR",
                 "Even fractions let credible leaders trade short.", limit=2)
        add_rows(is_prom, "Back-to-Lay",
                 "Back pre-off (prominent)",
                 "Lay @ ~0.55–0.75× entry IR",
                 "Prominent stalkers sit in the sweet spot in Even races.")
        add_rows(is_hold, "Back-late",
                 "No pre-off bet",
                 "Exit if halves to ~0.5× late",
                 "Closers need late pace; only back reactively.", limit=1)

    elif scenario == "Strong":
        add_rows(is_front_h, "Back-to-Lay (early)",
                 "Back pre-off for early dash",
                 "Lay @ 0.2–0.4× entry by 1–1.5f out",
                 "Leaders blaze and trade very short before tiring.", limit=2)
        add_rows(is_prom, "Lay-to-Back",
                 "Small lay pre-off (chasing heat)",
                 "Back @ 1.5–2.5× entry IR",
                 "Prominent chasers flatten behind multiple pace sources.", limit=3)
        add_rows(is_mid, "Back-late / Cover",
                 "Watch-only pre-off",
                 "Lay out if halves late",
                 "Mids benefit when leaders overdo it; reactive entry.", limit=2)

    else:  # Very Strong
        add_rows(is_front_h | is_front_q, "Lay-to-Back",
                 "Lay pre-off (pace meltdown risk)",
                 "Back @ 3–6× entry IR",
                 "Very Strong pace punishes leaders.", limit=3)
        add_rows(is_mid | is_hold, "Back-late",
                 "No pre-off bet",
                 "Lay out on 0.4–0.6× move",
                 "Second-wave mids/closers thrive in meltdowns.", limit=3)

    out = pd.DataFrame(plays)
    if out.empty:
        return out
    # Sort by play then suitability / class edge
    order = {"Back-to-Lay": 0, "Back-to-Lay (early)": 0, "Lay-to-Back": 1,
             "Back-late": 2, "Back-late / Cover": 2}
    out["_ord"] = out["Play"].map(order).fillna(3)
    out = out.sort_values(["_ord", "Suitability", "ΔvsPar"],
                          ascending=[True, False, False]).drop(columns=["_ord"]).reset_index(drop=True)
    return out

# -----------------------------
# Input Normalization
# -----------------------------

def normalize_two_files(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    REQUIRED = ["Horse","RS_Lto1","RS_Lto2","RS_Lto3","RS_Lto4","RS_Lto5","AdjSpeed"]
    aliases = {
        "horse": "Horse", "horse name": "Horse", "horse_name": "Horse", "name": "Horse", "runner": "Horse",
        "adj_speed": "AdjSpeed", "speed": "AdjSpeed", "key_speed": "AdjSpeed", "key speed factors average": "AdjSpeed",
        "lto1": "RS_Lto1","lto2":"RS_Lto2","lto3":"RS_Lto3","lto4":"RS_Lto4","lto5":"RS_Lto5",
        "rs1":"RS_Lto1","rs2":"RS_Lto2","rs3":"RS_Lto3","rs4":"RS_Lto4","rs5":"RS_Lto5",
    }

    def std(df):
        cols = {c: aliases.get(str(c).strip().lower(), c) for c in df.columns}
        out = df.rename(columns=cols).copy()
        if "Horse" in out.columns:
            out["Horse"] = out["Horse"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        return out

    A, B = std(df_a), std(df_b)
    merged = pd.merge(A, B, on="Horse", how="left")
    for col in REQUIRED:
        if col not in merged.columns:
            merged[col] = None
    for c in [f"RS_Lto{i}" for i in range(1,6)]:
        merged[c] = merged[c].fillna(0).astype(int)
    info = f"Normalized columns: {list(merged.columns)}. Rows: {len(merged)}"
    return merged[REQUIRED], info

# -----------------------------
# Utility: Build HorseRows
# -----------------------------

def to_rows(df: pd.DataFrame) -> List[HorseRow]:
    rows = []
    for _, r in df.iterrows():
        styles = [int(r.get(f"RS_Lto{i}", 0)) for i in range(1, 6)]
        adj = None if pd.isna(r.get("AdjSpeed")) else float(r["AdjSpeed"])
        rows.append(HorseRow(horse=str(r["Horse"]), run_styles=styles, adj_speed=adj))
    return rows

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="PaceForm", page_icon="🏇", layout="wide")
st.title("🏇 PaceForm – Pace & Class Analysis")

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

st.markdown("#### Upload input files (both required)")
left, right = st.columns(2)
with left:
    f1 = st.file_uploader("File A (horses & run styles)", type=["csv"], key="file_a")
with right:
    f2 = st.file_uploader("File B (speed figures)", type=["csv"], key="file_b")

if not (f1 and f2):
    st.info("Please upload both CSV files to generate the analysis.")
    st.stop()

try:
    df, info = normalize_two_files(pd.read_csv(f1), pd.read_csv(f2))
    st.success("Files normalized ✔")
    st.caption(info)
except Exception as e:
    st.error(f"Failed to read/normalize files: {e}")
    st.stop()

rows = to_rows(df)
res, debug = suitability(rows, s)
if res.empty:
    st.warning("No analysable rows. Check your files.")
    st.stop()

scenario = res.iloc[0]["Scenario"]
conf = float(res.iloc[0]["Confidence"]) if "Confidence" in res.columns else 0.0

st.subheader(f"Projected Pace: {scenario} (confidence {conf:.2f})")

with st.expander("Why this pace?", expanded=True):
    st.write(debug)

# --- Main ratings table
st.markdown("### Suitability Ratings")
st.dataframe(res[["Horse","Style","ΔvsPar","LCP","PaceFit","SpeedFit","wp","ws","Suitability"]], use_container_width=True)

# --- Trading suggestions section
st.markdown("### Trading Suggestions (pace-driven)")
sug = trading_suggestions(res, debug)
if sug is None or sug.empty:
    st.info("No trading suggestions for this setup.")
else:
    st.dataframe(
        sug[["Horse","Style","LCP","ΔvsPar","Suitability","Play","Entry","Exit","Why"]],
        use_container_width=True,
        hide_index=True,
    )
    # Optional: allow download of suggestions
    st.download_button(
        "Download trading suggestions CSV",
        data=sug.to_csv(index=False).encode("utf-8"),
        file_name="paceform_trading_suggestions.csv",
        mime="text/csv",
    )
    st.caption("Targets are expressed as in-running multiples relative to your entry price "
               "(e.g., lay @ 0.6× entry = 40% compression). Adjust to liquidity.")

# --- Final verdict (top 3 by Suitability, tie-break SpeedFit)
st.markdown("### Final Verdict")
short = res.sort_values(["Suitability", "SpeedFit"], ascending=False).head(3)[[
    "Horse","Suitability","Style","ΔvsPar"
]]
for i, row in short.reset_index(drop=True).iterrows():
    medals = ["🥇","🥈","🥉"][i]
    st.write(f"{medals} **{row['Horse']}** – Score {row['Suitability']} | {row['Style']} | ΔvsPar {row['ΔvsPar']}")

# --- Download all results
csv_bytes = res.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv_bytes,
    file_name="paceform_results.csv",
    mime="text/csv",
)

st.caption("Front-aware model with sprint-aware (5f/6f) handling, no-front & dominant-front caps, single-front cap, "
           "weak solo leader, and penalties for missing class figures (adaptive speed weight) + pace-driven trading suggestions.")
