import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

PITCH_CSV = "Trackman_Database_CLEANED_FINAL.csv"

RUN_VALUE_COL = "xRV of Event (Count Ind)"

#These are outcomes of a pitch being thrown, the call (ball or strike), and the swing (Y/N)
#These are two of the most significant factors in grading a pitcher
#Generally the more K's a pitcher gets the better
#But they should NOT be prioritized when grading a pitch soly based on it's properties (Spin Rate, Speed, etc.)
EVENT_SOURCE_COL = "PitchCall"
CSW_EVENTS = ["StrikeCalled", "StrikeSwinging"]

#Featured Columns # These are the columns that are used to train the models.
FEATURE_COLS = {
    "RelSpeed": "RelSpeed",
    "RelHeight": "RelHeight",
    "RelSide": "RelSide",
    "Extension": "Extension",
    "InducedVertBreak": "InducedVertBreak",
    "HorzBreak": "HorzBreak",
    "HorzApprAngle": "HorzApprAngle",
}

DELTA_FEATURES = ["HorzBreak", "InducedVertBreak", "RelSpeed",
                  "VertApprAngle", "HorzApprAngle"]

INTERACTION_PAIRS = [
    ("RelSpeed", "InducedVertBreak"),
    ("InducedVertBreak", "HorzBreak"),
    ("SpinRate", "InducedVertBreak"),
    ("RelSpeed", "VertApprAngle"),
]

VAA_COL = "VertApprAngle"
REL_HEIGHT_COL = "RelHeight"

PITCHER_HAND_COL = "PitcherThrows"
BATTER_HAND_COL = "BatterSide"
HAND_SIGNED_FEATURES = ["HorzBreak", "RelSide", "HorzApprAngle"]
PITCHER_ID_COL = "Pitcher"
PITCH_TYPE_COL = "CorrectedPitchType"

N_SPLITS = 5
RANDOM_STATE = 42
SCALE_MEAN = 100.0
SCALE_SD = 10.0
MIN_PITCHES_PER_GROUP = 50
MIN_PITCHES_PER_PITCHER = 200
TOP_N_PRINT = 10

OUT_DIR = "pitch_grade_output"

LGB_PARAMS_RV = {
    "learning_rate": 0.01,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

LGB_PARAMS_CSW = {
    "learning_rate": 0.01,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


def info(msg): print(f"[info] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)
def fail(msg): print(f"[FATAL] {msg}", flush=True); sys.exit(1)


def hand_sign(series):
    s = series.astype(str).str.strip().str.lower()
    return np.where(s.isin(["right", "r"]), 1.0,
                    np.where(s.isin(["left", "l"]), -1.0, 0.0))


def safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_") or "unknown"


def height_adjusted_vaa(df):
    if VAA_COL not in df.columns or REL_HEIGHT_COL not in df.columns:
        return None
    v = pd.to_numeric(df[VAA_COL], errors="coerce")
    h = pd.to_numeric(df[REL_HEIGHT_COL], errors="coerce")
    mask = v.notna() & h.notna()
    if mask.sum() < 100:
        return None
    slope, intercept = np.polyfit(h[mask].values, v[mask].values, 1)
    resid = v - (slope * h + intercept)
    info(f"Height-adjusted VAA: slope={slope:.4f} intercept={intercept:.4f}")
    return resid


def load_and_prepare():
    if not os.path.exists(PITCH_CSV):
        fail(f"Pitch CSV not found at '{PITCH_CSV}'.")
    info(f"Loading {PITCH_CSV} ...")
    df = pd.read_csv(PITCH_CSV, low_memory=False)
    info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    if RUN_VALUE_COL not in df.columns:
        fail(f"RUN_VALUE_COL '{RUN_VALUE_COL}' not in CSV.")
    df["run_value"] = pd.to_numeric(df[RUN_VALUE_COL], errors="coerce")

    if EVENT_SOURCE_COL not in df.columns:
        fail(f"EVENT_SOURCE_COL '{EVENT_SOURCE_COL}' not in CSV.")
    ev = df[EVENT_SOURCE_COL].astype(str).str.strip()
    if not any(e in set(ev.unique()) for e in CSW_EVENTS):
        warn(f"None of CSW_EVENTS {CSW_EVENTS} found. Seen: {sorted(set(ev.unique()))[:15]}")
    df["csw"] = ev.isin(CSW_EVENTS).astype(int)

    df = df.dropna(subset=["run_value"]).reset_index(drop=True)

    feature_names = []
    present = {f: c for f, c in FEATURE_COLS.items() if c in df.columns}
    for f in FEATURE_COLS:
        if f not in present:
            warn(f"Feature '{FEATURE_COLS[f]}' missing -- skipping.")
    for f, c in present.items():
        df[f] = pd.to_numeric(df[c], errors="coerce")
        feature_names.append(f)
    if not feature_names:
        fail("No feature columns found.")

    if PITCHER_HAND_COL in df.columns:
        psign = hand_sign(df[PITCHER_HAND_COL])
        for f in HAND_SIGNED_FEATURES:
            if f in df.columns:
                df[f] = df[f] * psign
    else:
        warn(f"PITCHER_HAND_COL '{PITCHER_HAND_COL}' missing.")

    new_cols = {}

    resid = height_adjusted_vaa(df)
    if resid is not None:
        new_cols["VertApprAngle_hadj"] = resid.values
        feature_names.append("VertApprAngle_hadj")

    has_type = PITCH_TYPE_COL in df.columns
    has_pitcher = PITCHER_ID_COL in df.columns
    for f in DELTA_FEATURES:
        if f not in df.columns:
            warn(f"Delta base feature '{f}' missing -- skipping its delta.")
            continue
        if has_type and has_pitcher:
            self_mean = df.groupby([PITCHER_ID_COL, PITCH_TYPE_COL])[f].transform("mean")
            new_cols[f"{f}_vs_self"] = (df[f] - self_mean).values
            feature_names.append(f"{f}_vs_self")
        else:
            warn(f"Cannot build '{f}_vs_self' (need pitcher and pitch-type columns).")

    for a, b in INTERACTION_PAIRS:
        if a in df.columns and b in df.columns:
            name = f"{a}_x_{b}"
            new_cols[name] = (df[a] * df[b]).values
            feature_names.append(name)
        else:
            warn(f"Interaction ({a}, {b}) skipped -- a base feature is missing.")

    if BATTER_HAND_COL in df.columns:
        new_cols["batter_is_rhh"] = (hand_sign(df[BATTER_HAND_COL]) > 0).astype(int)
        feature_names.append("batter_is_rhh")
    else:
        warn(f"BATTER_HAND_COL '{BATTER_HAND_COL}' missing.")

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()

    if "RelSpeed" in df.columns:
        df = df.dropna(subset=["RelSpeed"]).reset_index(drop=True)

    info(f"Final feature set ({len(feature_names)}): {feature_names}")
    info(f"Overall CSW rate: {df['csw'].mean():.3%}")
    return df, feature_names

df, feature_names = load_and_prepare()

df = pd.DataFrame(df)

df = df.select_dtypes(include='number')

df = df.drop(columns=["PitchNo", "PAofInning", "PitchofPA", "PitcherId"])
print(df)

pearson_corr = df.corr(method='pearson').to_csv('correlation_matrix.csv')
# print(pearson_corr)

