#VERSION 2C
#Reason: 2B would not run on anyones computer
#PLAN: Change one thing -> Run Program, make "version2___.py", repeat plan.
#Adding comments helps people see what's going on :) 
from __future__ import annotations

#import stuff
import os
import sys
import json
import warnings

#import more stuff
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from datetime import datetime

# 1. Get the current date and time formatted for filenames
# Example format: 2026-10-13_14-09-00
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

warnings.simplefilter("ignore", category=FutureWarning)

#Allegedly, this .csv file also goes by the name ~Austin Trackman Data, 
#but we do not know for sure if they are the same
PITCH_CSV = "Trackman_Database_CLEANED_FINAL.csv"

RUN_VALUE_COL = "xRV of Event (Count Ind)"

#These are outcomes of a pitch being thrown, the call (ball or strike), and the swing (Y/N)
#These are two of the most significant factors in grading a pitcher
#Generally the more K's a pitcher gets the better
#But they should NOT be prioritized when grading a pitch soly based on it's properties (Spin Rate, Speed, etc.)
EVENT_SOURCE_COL = "PitchCall"
CSW_EVENTS = ["StrikeCalled", "StrikeSwinging"]

#Featured Columns
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
    ("HorzBreak", "InducedVertBreak"),
    ("HorzBreak", "RelSpeed"),
    ("HorzBreak", "VertApprAngle"),
    ("HorzBreak", "HorzApprAngle"),
    ("InducedVertBreak", "RelSpeed"),
    ("InducedVertBreak", "VertApprAngle"),
    ("InducedVertBreak", "HorzApprAngle"),
    ("RelSpeed", "VertApprAngle"),
    ("RelSpeed", "HorzApprAngle"),
    ("VertApprAngle", "HorzApprAngle")
]

# INTERACTION_PAIRS = [
#     ("RelSpeed", "InducedVertBreak"),
#     ("InducedVertBreak", "HorzBreak"),
#     ("SpinRate", "InducedVertBreak"),
#     ("RelSpeed", "VertApprAngle"),
# ]

VAA_COL = "VertApprAngle"
REL_HEIGHT_COL = "RelHeight"

PITCHER_HAND_COL = "PitcherThrows" #Left or Right
BATTER_HAND_COL = "BatterSide" #Left or Right

# e.g (e.g. arm-side run becomes the same direction for both).
HAND_SIGNED_FEATURES = ["HorzBreak", "RelSide", "HorzApprAngle"] 
PITCHER_ID_COL = "Pitcher" # pitcher name / ID Column
PITCH_TYPE_COL = "CorrectedPitchType" #fastball, slider, etc.
#Q How are they determining this??

N_SPLITS = 5 # Number of cross-validation folds
RANDOM_STATE = 42 # Fixed seed so results are repeatable
SCALE_MEAN = 100.0 # Final grades, 100 being exactly average
SCALE_SD = 10.0 # 10 points is 1 Standard deviation

# This means that a pitchers pitch type needs to 
# have been thrown at least 50 times to appear in the leaders board
MIN_PITCHES_PER_GROUP = 50

# And pitchers need 200 pitch minimum to be considered in Overall leaderboard
# i.e. If Jane Doe throws 8 pitches, the worlds best curveball in history, then best slider, etc.
# She will not appear in any boards
# But her pitches will still have scores ~145.
MIN_PITCHES_PER_PITCHER = 200

# Rows for each leader board
TOP_N_PRINT = 10

# Folder where all the results get saved
OUT_DIR = f"{timestamp}_pitch_grade_output"

# Our settings for the Stuff+ Model
LGB_PARAMS_RV = {
    "learning_rate": 0.01,    # small steps = slow but careful learning
    "num_leaves": 63,    # how complex each tree can become
    "min_child_samples": 200,    # each leaf of tree needs >= 200 pitches
    "subsample": 0.8,    # each tree sees a random 80% of rows
    "subsample_freq": 1,    # each tree gets re-picked??
    "colsample_bytree": 0.8,    # each tree sees a random 80% of features
    "reg_lambda": 1.0,    # regularization
    "n_estimators": 3000, # Our MAX number of trees
    "random_state": RANDOM_STATE, # "Repeatable randomness" whatever that means
    "n_jobs": -1, #use of all CPU Cores"
    "verbose": -1,    #No lightGBM Chatter 
}

# These are the light gradient boosting parameters for CSW
# They are currently (from the summer to 10/22/26) the same as the RV parameters
# Which may be leading to some issues
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

# Now begins the list of def statements

#Tiny helpers for printing messages with a tag in front
def info(msg): print(f"[info] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)
    # fail() prints an ERROR and stops the entire program
def fail(msg): print(f"[FATAL] {msg}", flush=True); sys.exit(1)

#Turns a handedness column into numbers:
# Right ("right" or "r") -> +1, Left ("left" or "l") -> -1, anything else/missing -> 0
# Works no matter the capitalization or extra spaces
def hand_sign(series):
    s = series.astype(str).str.strip().str.lower()
    return np.where(s.isin(["right", "r"]), 1.0,
                    np.where(s.isin(["left", "l"]), -1.0, 0.0))

# Cleans text so it is safe to use a file name
# (every character that isn't a letter/number becomes "_"), e.g. "Two_Seam_FB" -> "Two_Seam_FB"
def safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_") or "unknown"

# the Vertical Apporach Angle depends a lot on how high the ball was released
# So this removes that effect:
#     1. Draw a best-fit straight line of VAA vs. Release Height
#     2. Return how far each pitch's VAA is ABOVE/BELOW that line (The "residual")
# Returns none if the columns are missing or there are fewer than 100 usuable rows
def height_adjusted_vaa(df):
    if VAA_COL not in df.columns or REL_HEIGHT_COL not in df.columns:
        return None
    v = pd.to_numeric(df[VAA_COL], errors="coerce")
    h = pd.to_numeric(df[REL_HEIGHT_COL], errors="coerce")
    mask = v.notna() & h.notna() # only use rows where both values exist
    if mask.sum() < 100:
        return None
    slope, intercept = np.polyfit(h[mask].values, v[mask].values, 1) # degree 1 = straight line
    resid = v - (slope * h + intercept)
    info(f"Height-adjusted VAA: slope={slope:.4f} intercept={intercept:.4f}")
    return resid

# Reads the CSV and builds every column the model needs
# Returns the cleaned table and list of the feature names the models will use
def load_and_prepare():
    # Load the file:
    if not os.path.exists(PITCH_CSV):
        fail(f"Pitch CSV not found at '{PITCH_CSV}'.")
    info(f"Loading {PITCH_CSV} ...")
    df = pd.read_csv(PITCH_CSV, low_memory=False)
    info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # Target 1: run values (This is what Stuff+ predicts)
    if RUN_VALUE_COL not in df.columns:
        fail(f"RUN_VALUE_COL '{RUN_VALUE_COL}' not in CSV.")
    df["run_value"] = pd.to_numeric(df[RUN_VALUE_COL], errors="coerce")

    # Target 2: CSW (this is what CSW+ predicts)
    # CSW is a 1 if the pitch was called a strike
    # (Meaning both the pitcher threw in the strike zone...
    # ...and the batter either swung & missed or didn't swing ...
    # ...OR the pitcher threw a out the strike zone, but the batter still swung
    # Otherwise -> 0
    if EVENT_SOURCE_COL not in df.columns:
        fail(f"EVENT_SOURCE_COL '{EVENT_SOURCE_COL}' not in CSV.")
    ev = df[EVENT_SOURCE_COL].astype(str).str.strip()
    if not any(e in set(ev.unique()) for e in CSW_EVENTS):
        warn(f"None of CSW_EVENTS {CSW_EVENTS} found. Seen: {sorted(set(ev.unique()))[:15]}")
    df["csw"] = ev.isin(CSW_EVENTS).astype(int)

    df = df.dropna(subset=["run_value"]).reset_index(drop=True)

    # Base features: copy in each FEATURE_COLS column as a number
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

    # This flips lefty movement and location so that lefties and righties are comparable
    # Like mirror mode on an iphone picture (I think??)
    if PITCHER_HAND_COL in df.columns:
        psign = hand_sign(df[PITCHER_HAND_COL])
        for f in HAND_SIGNED_FEATURES:
            if f in df.columns:
                df[f] = df[f] * psign
    else:
        warn(f"PITCHER_HAND_COL '{PITCHER_HAND_COL}' missing.")

    new_cols = {}

    # Height adjusted VAA
    resid = height_adjusted_vaa(df)
    if resid is not None:
        new_cols["VertApprAngle_hadj"] = resid.values
        feature_names.append("VertApprAngle_hadj")

    # This is for the "vs self" feature:
    # This pitch minus the pitchers own average for that single pitch type
    has_type = PITCH_TYPE_COL in df.columns
    has_pitcher = PITCHER_ID_COL in df.columns
    for f in DELTA_FEATURES:
        if f not in df.columns:
            warn(f"Delta base feature '{f}' missing -- skipping its delta.")
            continue
        if has_type and has_pitcher:
            # This will give each combination of pitcher and pitch type its own average (pitcher, pitch type)
            self_mean = df.groupby([PITCHER_ID_COL, PITCH_TYPE_COL])[f].transform("mean")
            new_cols[f"{f}_vs_self"] = (df[f] - self_mean).values
            feature_names.append(f"{f}_vs_self")
        else:
            warn(f"Cannot build '{f}_vs_self' (need pitcher and pitch-type columns).")

    # Interaction features: mutliply each pair from INTERACTION PAIRS
    for a, b in INTERACTION_PAIRS:
        if a in df.columns and b in df.columns:
            name = f"{a}_x_{b}"
            new_cols[name] = (df[a] * df[b]).values
            feature_names.append(name)
        else:
            warn(f"Interaction ({a}, {b}) skipped -- a base feature is missing.")

    # 1 if the batter is right handed, 0 if the batter is left
    if BATTER_HAND_COL in df.columns:
        new_cols["batter_is_rhh"] = (hand_sign(df[BATTER_HAND_COL]) > 0).astype(int)
        feature_names.append("batter_is_rhh")
    else:
        warn(f"BATTER_HAND_COL '{BATTER_HAND_COL}' missing.")

    # Attach all the new columns to the table in one go
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()

    # Pitches with no velocity are dropped (speed is the most basic feature)
    if "RelSpeed" in df.columns:
        df = df.dropna(subset=["RelSpeed"]).reset_index(drop=True)

    info(f"Final feature set ({len(feature_names)}): {feature_names}")
    info(f"Overall CSW rate: {df['csw'].mean():.3%}")
    return df, feature_names

# Builds the cross-validation folds.
# GroupKFold keeps ALL of a pitcher's pitches in the same fold, so the model is never
# validated on a pitcher it already saw in training (that would make scores look too good).
def get_folds(df):
    if PITCHER_ID_COL in df.columns:
        groups = df[PITCHER_ID_COL].astype(str).values
    else:
        warn(f"PITCHER_ID_COL '{PITCHER_ID_COL}' missing -- pseudo groups.")
        groups = (np.arange(len(df)) % (N_SPLITS * 50)).astype(str)
    return list(GroupKFold(n_splits=N_SPLITS).split(df, df["run_value"], groups))

# Figures out HOW MANY TREES to use.
# For each fold: train with early stopping (quit once the validation score hasn't improved
# for 150 trees) and record the best tree count. Returns the median across folds.
#   is_clf=True  -> classifier (CSW), judged by AUC
#   is_clf=False -> regressor (run value), judged by RMSE
def median_best_iter(X, y, folds, is_clf, params):
    Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
    obj = "binary" if is_clf else "regression"
    metric = "auc" if is_clf else "rmse"
    iters = []
    for tr, va in folds:
        m = Est(objective=obj, **params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              eval_metric=metric,
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])
        iters.append(int(m.best_iteration_ or params["n_estimators"]))
    return int(np.median(iters))



# This is what actives when you're trying to get on a train but its full
# Trains the FINAL model on ALL the data, using the tree count found by median_best_iter
def train_full(X, y, n_estimators, is_clf, params):
    Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
    obj = "binary" if is_clf else "regression"
    p = dict(params)
    p["n_estimators"] = n_estimators
    m = Est(objective=obj, **p)
    m.fit(X, y)
    return m

# Converts raw model predictions into the 100-centered grade scale.
#   invert=True  flips the sign first. Used for run value, where we assume a LOWER
#                predicted run value is better for the pitcher, so it should score higher.
#   z-score: (value - average) / standard deviation, then rescaled so average = 100, 1 SD = 10
#   (the 1e-12 just prevents dividing by zero)
def scale_grade(pred, invert):
    v = -np.asarray(pred) if invert else np.asarray(pred)
    z = (v - v.mean()) / (v.std(ddof=0) + 1e-12)
    return SCALE_MEAN + SCALE_SD * z

# Saves a bar chart showing which features the model relied on most
# ("gain" = how much each feature improved the model when it was used in a split)
def save_importance_chart(model, feature_names, target_col, label):
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                    index=feature_names).sort_values()
    plt.figure(figsize=(9, max(3, 0.35 * len(imp))))
    imp.plot(kind="barh")
    plt.title(f"Feature importance (gain) -- {label}")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, target_col, f"feature_importance_{target_col}.png")
    plt.savefig(path, dpi=130)
    plt.close()
    info(f"{label}: saved feature importance chart -> {path}")

# Makes a heatmap of which features matter most FOR EACH PITCH TYPE
# (e.g. velocity may matter most for fastballs, break for sliders).
# Uses SHAP values: how much each feature pushed each pitch's prediction up or down.
def save_per_pitchtype_importance(model, X, pitch_types, feature_names, target_col, label):
    if pitch_types is None:
        warn(f"{label}: no pitch-type column -- skipping per-pitch-type importance.")
        return
    # pred_contrib=True returns one SHAP value per feature per pitch, plus an extra
    # final "baseline" column. [:, :len(feature_names)] drops that extra column.
    contribs = np.asarray(model.booster_.predict(X, pred_contrib=True))[:, :len(feature_names)]
    abs_df = pd.DataFrame(np.abs(contribs), columns=feature_names)
    abs_df["_pt"] = np.asarray(pitch_types)

     # Only keep pitch types with enough pitches to be meaningful
    counts = abs_df["_pt"].value_counts()
    keep_types = counts[counts >= MIN_PITCHES_PER_GROUP].index.tolist()
    abs_df = abs_df[abs_df["_pt"].isin(keep_types)]
    if not keep_types:
        warn(f"{label}: no pitch types clear the sample threshold -- skipping heatmap.")
        return

    mat = abs_df.groupby("_pt").mean().T
    order = mat.mean(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order]
    norm = mat / (mat.sum(axis=0) + 1e-12)

    sub = os.path.join(OUT_DIR, target_col)
    mat.to_csv(os.path.join(sub, f"pitchtype_importance_raw_{target_col}.csv"))
    norm.to_csv(os.path.join(sub, f"pitchtype_importance_share_{target_col}.csv"))

    # Draw the heatmap
    fig, ax = plt.subplots(figsize=(1.3 * norm.shape[1] + 4, 0.4 * norm.shape[0] + 2))
    im = ax.imshow(norm.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(norm.shape[1]))
    ax.set_xticklabels(norm.columns, rotation=45, ha="right")
    ax.set_yticks(range(norm.shape[0]))
    ax.set_yticklabels(norm.index)
    ax.set_title(f"Feature importance share by pitch type -- {label}")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm.values[i, j] < norm.values.max() * 0.6 else "black",
                    fontsize=7)
    fig.colorbar(im, ax=ax, label="share of |SHAP| within pitch type")
    plt.tight_layout()
    path = os.path.join(sub, f"pitchtype_importance_{target_col}.png")
    plt.savefig(path, dpi=130)
    plt.close()
    info(f"{label}: saved per-pitch-type importance heatmap -> {path}")


# Runs the whole process for ONE target (run value OR csw):
#   find best tree count -> train final model -> predict -> scale to grades -> save model + charts
# Returns one grade per pitch.
def score_target(df, feature_names, folds, target_col, is_clf, invert, label, params):
    info(f"=== {label} ===")
    X = df[feature_names]
    y = df[target_col].astype(int if is_clf else float)
    n = median_best_iter(X, y, folds, is_clf, params)
    info(f"{label}: median best_iter={n} (cap {params['n_estimators']}); refitting on full data ...")
    if n >= params["n_estimators"] - 1:
        warn(f"{label}: best_iter hit the n_estimators cap -- consider raising it further.")
    m = train_full(X, y, n, is_clf, params)
    pred = m.predict_proba(X)[:, 1] if is_clf else m.predict(X)
    grade = scale_grade(pred, invert)
    sub = os.path.join(OUT_DIR, target_col)
    os.makedirs(sub, exist_ok=True)
    m.booster_.save_model(os.path.join(sub, f"lightgbm_{target_col}.txt"))
    save_importance_chart(m, feature_names, target_col, label)
    pt = df[PITCH_TYPE_COL] if PITCH_TYPE_COL in df.columns else None
    save_per_pitchtype_importance(m, X, pt, feature_names, target_col, label)
    return grade

# Builds one leaderboard per pitch type (all fastballs, all sliders, etc.),
# prints the top N, and saves a CSV for each plus one combined CSV.
# `grp` = one row per (pitcher, pitch type) with the pitch count and average grades.
def build_pitchtype_leaderboards(grp):
    bt_dir = os.path.join(OUT_DIR, "leaderboards_by_pitchtype")
    os.makedirs(bt_dir, exist_ok=True)
    cols = [PITCHER_ID_COL, "pitches", "stuff_plus", "csw_plus"]

    combined_rows = []
    for pt, g in grp.groupby(PITCH_TYPE_COL):
        g_stuff = g.sort_values("stuff_plus", ascending=False)[cols].reset_index(drop=True)
        g_stuff.insert(0, PITCH_TYPE_COL, pt)
        g_stuff.insert(1, "rank_stuff", np.arange(1, len(g_stuff) + 1))
        g_csw_rank = (g.sort_values("csw_plus", ascending=False)[PITCHER_ID_COL]
                       .reset_index(drop=True))
        csw_rank_map = {p: i + 1 for i, p in enumerate(g_csw_rank)}
        g_stuff["rank_csw"] = g_stuff[PITCHER_ID_COL].map(csw_rank_map)
        g_stuff.to_csv(os.path.join(bt_dir, f"{safe_name(pt)}.csv"), index=False)
        combined_rows.append(g_stuff)

        print(f"\nTop {TOP_N_PRINT} '{pt}' by Stuff+ (n={len(g)}):", flush=True)
        print(g_stuff.head(TOP_N_PRINT)[[PITCHER_ID_COL, "pitches", "stuff_plus",
                                         "csw_plus"]].to_string(index=False), flush=True)
        print(f"Top {TOP_N_PRINT} '{pt}' by CSW+:", flush=True)
        print(g.sort_values("csw_plus", ascending=False).head(TOP_N_PRINT)[
            [PITCHER_ID_COL, "pitches", "stuff_plus", "csw_plus"]].to_string(index=False),
            flush=True)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_path = os.path.join(OUT_DIR, "leaderboards_by_pitchtype_all.csv")
        combined.to_csv(combined_path, index=False)
        info(f"Saved per-pitch-type leaderboards -> {bt_dir}\\ and {combined_path}")

# THE MAIN PROGRAM: everything above gets called from here, in order
def main():
    os.makedirs(OUT_DIR, exist_ok=True)   # make the timestamped output folder
    print("=" * 70, flush=True)
    print(" Full Stuff model: interactions + self-deltas + height-adj VAA (RV + CSW%)", flush=True)
    print("=" * 70, flush=True)
 
    # Step 1: load data and build features
    df, feature_names = load_and_prepare()
    # Step 2: set up cross-validation folds (grouped by pitcher)
    folds = get_folds(df)
 
    # Step 3: train both models and get a grade for every pitch
    # invert=True for run value (lower run value = better for the pitcher)
    stuff_plus = score_target(df, feature_names, folds, "run_value",
                              is_clf=False, invert=True, label="Stuff+ (run value)",
                              params=LGB_PARAMS_RV)
    csw_plus = score_target(df, feature_names, folds, "csw",
                            is_clf=True, invert=False, label="CSW+",
                            params=LGB_PARAMS_CSW)
 
    # Step 4: save every pitch with its two grades
    scored = df[[c for c in [PITCHER_ID_COL, PITCH_TYPE_COL] if c in df.columns]].copy()
    scored["run_value"] = df["run_value"].values
    scored["csw"] = df["csw"].values
    scored["stuff_plus"] = stuff_plus
    scored["csw_plus"] = csw_plus
    scored.to_csv(os.path.join(OUT_DIR, "scored_pitches.csv"), index=False)
    info(f"Saved per-pitch scores -> {os.path.join(OUT_DIR, 'scored_pitches.csv')}")
 
    # Step 5: roll the per-pitch grades up into pitcher / pitch-type grades and leaderboards
    if PITCHER_ID_COL in df.columns and PITCH_TYPE_COL in df.columns:
        # One row per (pitcher, pitch type): pitch count and average grades
        grp = (scored.groupby([PITCHER_ID_COL, PITCH_TYPE_COL])
                     .agg(pitches=("stuff_plus", "size"),
                          stuff_plus=("stuff_plus", "mean"),
                          csw_plus=("csw_plus", "mean")).reset_index())
        grp = grp[grp["pitches"] >= MIN_PITCHES_PER_GROUP].copy()   # drop small samples
        grp["stuff_plus"] = grp["stuff_plus"].round(1)
        grp["csw_plus"] = grp["csw_plus"].round(1)
        grp = grp.sort_values([PITCHER_ID_COL, "stuff_plus"], ascending=[True, False])
        path = os.path.join(OUT_DIR, "pitcher_pitch_grades.csv")
        grp.to_csv(path, index=False)
        info(f"Saved per pitcher-pitch-type grades ({len(grp)} rows) -> {path}")
 
        # Average a column across a pitcher's pitch types, weighted by how often he throws each
        # (so a pitch he throws 500 times counts more than one he throws 50 times)
        def weighted(g, col):
            return float(np.average(g[col].values, weights=g["pitches"].values))
 
        # Overall leaderboard: one row per pitcher
        lb = (grp.groupby(PITCHER_ID_COL)
                 .apply(lambda g: pd.Series({
                     "pitches": int(g["pitches"].sum()),
                     "pitch_types": int(g[PITCH_TYPE_COL].nunique()),
                     "stuff_plus": weighted(g, "stuff_plus"),
                     "csw_plus": weighted(g, "csw_plus"),
                 }))
                 .reset_index())
        lb = lb[lb["pitches"] >= MIN_PITCHES_PER_PITCHER].copy()   # need enough total pitches
        lb["pitches"] = lb["pitches"].astype(int)
        lb["pitch_types"] = lb["pitch_types"].astype(int)
        lb["stuff_plus"] = lb["stuff_plus"].round(1)
        lb["csw_plus"] = lb["csw_plus"].round(1)
        lb = lb.sort_values("stuff_plus", ascending=False)
        lb_path = os.path.join(OUT_DIR, "pitcher_leaderboard.csv")
        lb.to_csv(lb_path, index=False)
        info(f"Saved overall pitcher leaderboard ({len(lb)} pitchers) -> {lb_path}")
 
        # Print the console leaderboards
        print("\nTop 25 pitchers by arsenal-weighted Stuff+ (min "
              f"{MIN_PITCHES_PER_PITCHER} pitches):", flush=True)
        print(lb.head(25).to_string(index=False), flush=True)
        print("\nTop 25 pitchers by arsenal-weighted CSW+:", flush=True)
        print(lb.sort_values("csw_plus", ascending=False).head(25).to_string(index=False),
              flush=True)
 
        print(f"\nTop 25 pitcher pitch-types by Stuff+ (min "
              f"{MIN_PITCHES_PER_GROUP} pitches):", flush=True)
        print(grp.sort_values("stuff_plus", ascending=False).head(25).to_string(index=False),
              flush=True)
        print("\nTop 25 pitcher pitch-types by CSW+:", flush=True)
        print(grp.sort_values("csw_plus", ascending=False).head(25).to_string(index=False),
              flush=True)
 
        # Separate leaderboard for each pitch type
        build_pitchtype_leaderboards(grp)
    else:
        warn("Cannot build per pitcher-pitch-type grades (missing pitcher or pitch-type column).")
 
    # Step 6: save a record of this run's settings (handy for comparing versions later)
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump({"n_pitches": int(len(df)), "features": feature_names,
                   "min_pitches_per_group": MIN_PITCHES_PER_GROUP,
                   "min_pitches_per_pitcher": MIN_PITCHES_PER_PITCHER,
                   "interaction_pairs": [f"{a}_x_{b}" for a, b in INTERACTION_PAIRS],
                   "rv_params": LGB_PARAMS_RV, "csw_params": LGB_PARAMS_CSW}, f, indent=2)
    print("\nDone. See the '%s' folder." % OUT_DIR, flush=True)
 
 
# Only runs main() when you run this file directly (not when another file imports it)
if __name__ == "__main__":
    main()
