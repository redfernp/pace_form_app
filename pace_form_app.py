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

# Default PaceFit (routes / generic)
PACEFIT = {
    "Slow":   {"Front": 5, "Prominent": 4, "Mid": 3, "Hold-up": 2},
    "Even":   {"Front": 4, "Prominent": 5, "Mid": 4, "Hold-up": 3},
    "Strong": {"Front": 2, "Prominent": 3, "Mid": 4, "Hold-up": 5},
    "Very Strong": {"Front": 1, "Prominent": 2, "Mid": 4, "Hold-up": 5},
}

# --- NEW: Even (Front-Controlled) map for routes
# Small uplift to Front; Prominent unchanged; mild trims to Mid/Hold-up
PACEFIT_EVEN_FC = {"Front": 4.5, "Prominent": 5.0, "Mid": 3.5, "Hold-up": 2.5}

# Sprint-specific PaceFit maps
# 5f: position >> power; 6f: still pace-positional but slightly more forgiving
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
    """Return distance band: '5f', '6f', or 'route'."""
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


# --- CHANGE #1: Penalise missing figure (None) ---
def speed_score(dvp: Optional[float]) -> float:
    """
    Map Δ vs Par to a 1..5 speed score.
    Missing figure (None) now penalised to 2.3 (was 3.0) so unproven class
    doesn't float to the top in Slow/route races.
    """
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
# Pace Projection (front-aware + caps + weak solo leader + dominant front + sprint caps)
# -----------------------------

def project_pace(rows: List[HorseRow], s: Settings) -> Tuple[str, float, Dict[str, str], Dict[str, object]]:
    """
    Front-aware pace model with realistic caps and a debug payload.

    Rules encoded:
    1) Early-energy index (Front > Prominent; Questionables discounted).
    2) NO-FRONT CAP: If no Fronts (or no *High* Fronts), cap at 'Even' unless
       ≥3 High Prominent with mean ΔvsPar ≥ -1 (near par).
    3) DOMINANT-FRONT CAP:
       If exactly one High Front with ΔvsPar ≥ +2 and ≤1 High Prominent,
       cap Strong/Very Strong to 'Even' (leader likely to dictate).
    4) SINGLE-FRONT CAP (modest edge):
       If exactly one High Front and its ΔvsPar ≤ +1, cap to 'Even'.
       If that lone Front has ΔvsPar ≤ -2 and there are ≤1 High Prominent → allow 'Slow'.
    5) WEAK SOLO LEADER:
       If exactly one Front in the whole field and its ΔvsPar ≤ -8, downgrade one category.
    6) SPRINT CAPS (NEW for 5f & 6f):
       Lone credible Front near/above par with limited pressers → cap Strong→Even at short trips.
    """
    debug = {"rules_applied": []}

    if not rows:
        return "N/A", 0.0, {}, {"error": "no rows", "rules_applied": []}

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
        return "N/A", 0.0, {}, {"error": "empty dataframe", "rules_applied": []}

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
        debug["rules_applied"].append("Base: ≥2 High Front → Strong")
    elif (n_front_high + n_prom_high) >= 3 and early_energy >= 3.2:
        scenario, conf = "Strong", 0.65
        debug["rules_applied"].append("Base: ≥3 High early & energy≥3.2 → Strong")
    elif early_energy >= 3.2:
        scenario, conf = "Strong", 0.60
        debug["rules_applied"].append("Base: energy≥3.2 → Strong")
    elif (n_front_high + n_prom_high) >= 2:
        scenario, conf = "Even", 0.60
        debug["rules_applied"].append("Base: ≥2 High early → Even")
    elif (n_front_high + n_prom_high) == 1 and (n_front_q + n_prom_q) >= 1:
        scenario, conf = "Even", 0.55
        debug["rules_applied"].append("Base: 1 High + some Questionable → Even")
    elif (n_front_high + n_prom_high) == 1:
        scenario, conf = "Even", 0.60
        debug["rules_applied"].append("Base: 1 High only → Even")
    elif (n_front_q + n_prom_q) >= 1:
        scenario, conf = "Slow", 0.60
        debug["rules_applied"].append("Base: only Questionables → Slow")
    else:
        scenario, conf = "Slow", 0.70
        debug["rules_applied"].append("Base: no credible early → Slow")

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
            debug["rules_applied"].append("No-front cap: no High Front (and no strong Prominent cluster) → cap to Even")

    # 3) DOMINANT-FRONT CAP
    if n_front_high == 1 and n_prom_high <= 1:
        try:
            lone_front_dvp = float(front_high["dvp"].iloc[0])
        except Exception:
            lone_front_dvp = None
        if (lone_front_dvp is not None) and (lone_front_dvp >= 2.0):
            if scenario in ("Strong", "Very Strong"):
                scenario, conf = "Even", max(conf, 0.65)
                debug["rules_applied"].append("Dominant-front cap: 1 High Front with ΔvsPar≥+2 and ≤1 High Prominent → Strong→Even")

    # 4) SINGLE-FRONT CAP (modest edge or below par)
    if n_front_high == 1:
        try:
            lone_front_dvp2 = float(front_high["dvp"].iloc[0])
        except Exception:
            lone_front_dvp2 = None

        if (lone_front_dvp2 is None) or (lone_front_dvp2 <= 1.0):
            if scenario in ("Strong", "Very Strong"):
                scenario, conf = "Even", max(conf, 0.60)
                debug["rules_applied"].append("Single-front cap: 1 High Front with ΔvsPar≤+1 → Strong→Even")

        if (lone_front_dvp2 is not None) and (lone_front_dvp2 <= -2.0) and (n_prom_high <= 1):
            if scenario == "Even":
                scenario, conf = "Slow", max(conf, 0.65)
                debug["rules_applied"].append("Single-front below par & little pressure → Even→Slow")

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
            debug["rules_applied"].append("Weak solo leader: single Front ΔvsPar≤-8 → downgrade one")

    # 5b) TWO-HIGH-FRONT SUSTAINABILITY (band-aware soft cap)
    band = _dist_band(getattr(s, "distance_f", 7.0))
    if n_front_high == 2:
        try:
            fh_dvps = front_high["dvp"].dropna().astype(float).tolist()
        except Exception:
            fh_dvps = []
        cond_has_two = (len(fh_dvps) == 2)
        if band == "5f":
            cond_efficient = cond_has_two and (min(fh_dvps) >= -3.0) and (max(fh_dvps) <= 1.0)
            cond_pressers  = (n_prom_high <= 1)
            cond_energy    = (early_energy <= 4.0)
        elif band == "6f":
            cond_efficient = cond_has_two and (min(fh_dvps) >= -2.0) and (max(fh_dvps) <= 0.0)
            cond_pressers  = (n_prom_high == 0)
            cond_energy    = (early_energy < 3.4)
        else:  # routes (≥7f)
            cond_efficient = cond_has_two and (min(fh_dvps) >= -1.0) and (max(fh_dvps) <= 0.0)
            cond_pressers  = (n_prom_high == 0)
            cond_energy    = (early_energy < 3.0)

        if cond_efficient and cond_pressers and cond_energy:
            if scenario in ("Strong", "Very Strong"):
                scenario, conf = "Even", 0.55
                debug["rules_applied"].append(
                    f"Two High Fronts {band}: efficient + low pressers + low energy → Strong/Very Strong → Even"
                )
        else:
            if scenario in ("Strong", "Very Strong"):
                tag = "fragile" if band in ("5f","6f") else "kept"
                debug["rules_applied"].append(
                    f"Two High Fronts {band}: conditions not all met → keep {scenario} ({tag})"
                )

    # 6) SPRINT CAPS (5f & 6f)
    if band in ("5f", "6f"):
        cap_prom_limit = 2
        energy_cap = 4.0 if band == "5f" else 3.6
        dvp_ok = -0.5 if band == "5f" else -1.0
        if n_front_high == 1 and n_prom_high <= cap_prom_limit:
            try:
                lf_dvp = float(front_high["dvp"].iloc[0])
            except Exception:
                lf_dvp = None
            if (lf_dvp is not None) and (lf_dvp >= dvp_ok) and (early_energy < energy_cap):
                if scenario in ("Strong", "Very Strong"):
                    scenario, conf = "Even", max(conf, 0.65)
                    debug["rules_applied"].append(
                        f"Sprint cap ({band}): 1 High Front (Δ≥{dvp_ok}) with ≤{cap_prom_limit} High "
                        f"Prominent & energy<{energy_cap} → Strong→Even"
                    )

    # --- NEW: Tag "Even (Front-Controlled)" at routes
    front_controlled = False
    scenario_label = scenario
    try:
        lf_dvp_fc = float(front_high["dvp"].iloc[0]) if n_front_high == 1 else None
    except Exception:
        lf_dvp_fc = None
    if band == "route" and scenario == "Even" and n_front_high == 1 and n_prom_high <= 1 and (lf_dvp_fc is not None) and (lf_dvp_fc >= -1.0):
        front_controlled = True
        scenario_label = "Even (Front-Controlled)"
        debug["rules_applied"].append("Front-controlled tag: 1 High Front, ≤1 High Prominent, route, ΔvsPar≥-1 → Even (Front-Controlled)")
    debug["front_controlled"] = bool(front_controlled)

    debug.update({
        "counts": {
            "Front_all": int(n_front),
            "Front_High": int(n_front_high),
            "Front_Questionable": int(n_front_q),
            "Prominent_High": int(n_prom_high),
            "Prominent_Questionable": int(n_prom_q),
        },
        "early_energy": float(early_energy),
        "distance_band": band,
    })

    lcp_map = dict(zip(d["horse"], d["lcp"]))
    return scenario_label, conf, lcp_map, debug

# -----------------------------
# Suitability Scoring
# -----------------------------

def suitability(rows: List[HorseRow], s: Settings) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not rows:
        return pd.DataFrame(), {"error": "no rows"}
    scenario, conf, lcp_map, debug = project_pace(rows, s)
    if scenario == "N/A":
        return pd.DataFrame(), {"error": "pace N/A"}

    # Choose PaceFit by distance band
    band = _dist_band(getattr(s, "distance_f", 7.0))
    if scenario.startswith("Even (Front-Controlled)"):
        if band == "route":
            pacefit_map = PACEFIT_EVEN_FC
        else:
            # outside routes, fall back to standard "Even"
            pacefit_map = PACEFIT_6F["Even"] if band == "6f" else (PACEFIT_5F["Even"] if band == "5f" else PACEFIT["Even"])
    elif band == "5f":
        pacefit_map = PACEFIT_5F[scenario]
    elif band == "6f":
        pacefit_map = PACEFIT_6F[scenario]
    else:
        pacefit_map = PACEFIT[scenario]

    # weights: pace matters more in sprints
    wp = s.wp_confident if scenario in ("Slow", "Very Strong") and conf >= 0.65 else s.wp_even
    if band == "5f":
        wp = max(wp, 0.60)
    elif band == "6f":
        wp = max(wp, 0.55)
    ws = 1 - wp

    # --- CHANGE #2: Adaptive ws when many runners have no figure ---
    missing_count = sum(1 for r in rows if r.adj_speed is None)
    missing_ratio = missing_count / max(len(rows), 1)
    if missing_ratio > 0.5:
        # Reduce influence of SpeedFit when >50% of field are unknowns
        ws *= 0.8
        wp = 1 - ws
        debug["rules_applied"] = debug.get("rules_applied", []) + [
            f"Adaptive speed weight: {missing_ratio:.0%} missing figures → ws*0.8"
        ]
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
    """
    Build trade plays using projected pace + style + LCP.
    Returns columns: Horse, Style, LCP, ΔvsPar, Suitability, Play, Entry, Exit, Why
    """
    if res.empty:
        return pd.DataFrame()
    scenario = str(res.iloc[0].get("Scenario", "Even"))
    # Treat any Even-like label (incl. Even (Front-Controlled)) as "Even" for plays
    if scenario.startswith("Even"):
        scenario = "Even"
    # conf = float(res.iloc[0].get("Confidence", 0.6))  # kept for future use
    # band = str(debug.get("distance_band", "route"))    # kept for future use

    df = res.copy()
    # Masks
    is_front_h = (df["Style"] == "Front") & (df["LCP"] == "High")
    is_front_q = (df["Style"] == "Front") & (df["LCP"].isin(["Questionable","Unlikely"]))
    is_prom    = (df["Style"] == "Prominent")
    is_mid     = (df["Style"] == "Mid")
    is_hold    = (df["Style"] == "Hold-up")

    plays: List[Dict[str, object]] = []

    def add_rows(mask, play, entry_note, exit_note, rationale, limit=3):
        sub = df[mask].sort_values(["Suitability","ΔvsPar"], ascending=False).head(limit)
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
    # Sort by play, then Suitability and class edge
    order = {"Back-to-Lay":0, "Back-to-Lay (early)":0, "Lay-to-Back":1, "Back-late":2, "Back-late / Cover":2}
    out["_ord"] = out["Play"].map(order).fillna(3)
    out = out.sort_values(["_ord","Suitability","ΔvsPar"], ascending=[True, False, False]).drop(columns=["_ord"]).reset_index(drop=True)
    return out

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
res, debug = suitability(rows, s)
if res.empty:
    st.warning("No analysable rows after normalization. Check your input files.")
    st.stop()

scenario = res.iloc[0]["Scenario"]
conf = float(res.iloc[0]["Confidence"]) if "Confidence" in res.columns else 0.0

st.subheader(f"Projected Pace: {scenario} (confidence {conf:.2f})")

# ---- Reason used (explanation block)
with st.expander("Why this pace? (Reason used)", expanded=True):
    counts = debug.get("counts", {})
    st.markdown(
        f"- **Front (all):** {counts.get('Front_all', 0)} &nbsp;&nbsp; "
        f"**Front (High):** {counts.get('Front_High', 0)} &nbsp;&nbsp; "
        f"**Front (Questionable):** {counts.get('Front_Questionable', 0)}  \n"
        f"- **Prominent (High):** {counts.get('Prominent_High', 0)} &nbsp;&nbsp; "
        f"**Prominent (Questionable):** {counts.get('Prominent_Questionable', 0)}  \n"
        f"- **Early energy:** {debug.get('early_energy', 0.0):.2f}  \n"
        f"- **Distance band:** {debug.get('distance_band', 'route')}  \n"
        f"- **Missing speed ratio:** {debug.get('missing_speed_ratio', 0.0):.2f}"
    )
    rules = debug.get("rules_applied", [])
    if rules:
        st.markdown("**Rules applied:**")
        for r in rules:
            st.write(f"• {r}")
    else:
        st.caption("No specific adjustments beyond base rule.")

cols = st.columns([1.6, 1.4])
with cols[0]:
    # --- Main ratings table (with AvgPace for manual analysis)
    st.markdown("### Suitability Ratings")
    _df_show = res.copy()
    _df_show["AvgPace"] = _df_show["AvgStyle"].round(2)
    cols_order = ["Horse","AvgPace","Style","ΔvsPar","LCP","PaceFit","SpeedFit","wp","ws","Suitability"]
    cols_order = [c for c in cols_order if c in _df_show.columns]
    st.dataframe(_df_show[cols_order], use_container_width=True)

with cols[1]:
    st.markdown("### Run Style Summary")
    show = res[["Horse","AvgStyle","Style","LCP","ΔvsPar"]].copy()
    st.dataframe(show, use_container_width=True)

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
    st.download_button(
        "Download trading suggestions CSV",
        data=sug.to_csv(index=False).encode("utf-8"),
        file_name="paceform_trading_suggestions.csv",
        mime="text/csv",
    )
    st.caption(
        "Targets are expressed as in-running multiples relative to your entry price "
        "(e.g., lay @ 0.6× entry = 40% compression). Adjust to liquidity."
    )

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

st.caption("Front-aware model with sprint-aware (5f/6f) handling, no-front & dominant-front caps, single-front cap, weak solo leader, and penalties for missing class figures (adaptive speed weight).")
