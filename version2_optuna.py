from __future__ import annotations

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna  # Added hyperparameter optimization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ( 
    mean_squared_error, 
    mean_absolute_error, 
    roc_auc_score, 
    log_loss)

warnings.simplefilter("ignore", category=FutureWarning)

# V2 + Optuna: separate RV/CSW tuning with pitcher-grouped CV
# Preserves V2 features, interactions, grading, and leaderboard outputs
# Fixes independent target filtering and missing CSW labels; raises the CSW tree cap

# CSV file containing pitch measurements and outcomes
PITCH_CSV = "austinTrackmanData.csv"

# Run value is the numeric outcome predicted by the RV model
RUN_VALUE_COL = "xRV of Event (Count Ind)"

# CSW means a called strike or a swinging strike
EVENT_SOURCE_COL = "PitchCall"
CSW_EVENTS = ["StrikeCalled", "StrikeSwinging"]

# Match model inputs to CSV columns and skip any missing inputs
FEATURE_COLS = {
    "RelSpeed": "RelSpeed",
    "RelHeight": "RelHeight",
    "RelSide": "RelSide",
    "Extension": "Extension",
    "InducedVertBreak": "InducedVertBreak",
    "HorzBreak": "HorzBreak",
    "HorzApprAngle": "HorzApprAngle",
}

# Compare these measurements with the same pitcher’s average for that pitch type
DELTA_FEATURES = ["HorzBreak", "InducedVertBreak", "RelSpeed",
                  "VertApprAngle", "HorzApprAngle"]

# Multiply each pair of measurements to create a combined input
INTERACTION_PAIRS = [
    ("RelSpeed", "InducedVertBreak"),
    ("InducedVertBreak", "HorzBreak"),
    ("SpinRate", "InducedVertBreak"),
    ("RelSpeed", "VertApprAngle"),
]

# Use release height to adjust the vertical approach angle
VAA_COL = "VertApprAngle"
REL_HEIGHT_COL = "RelHeight"

# Identify pitcher and batter handedness and the columns to mirror for left-handers
PITCHER_HAND_COL = "PitcherThrows"
BATTER_HAND_COL = "BatterSide"
HAND_SIGNED_FEATURES = ["HorzBreak", "RelSide", "HorzApprAngle"]
PITCHER_ID_COL = "Pitcher"
PITCH_TYPE_COL = "CorrectedPitchType"

# Set validation folds, repeatable randomness, grade scaling, and leaderboard limits
# Cross-validation trains and checks models on five groups of pitchers
N_SPLITS = 5
RANDOM_STATE = 42
# Grades average 100 with 10 points equal to one standard deviation
SCALE_MEAN = 100.0
SCALE_SD = 10.0
# Require enough valid pitches before showing a pitch-type or overall pitcher grade
MIN_PITCHES_PER_GROUP = 50
MIN_PITCHES_PER_PITCHER = 200
TOP_N_PRINT = 10

# Keep original V2 outputs separate
OUT_DIR = "pitch_grade_output_v2_optuna"

# Tune each target independently while preserving the V2 architecture
# Trial counts are total study budgets; completed trials are reused on restart
# Runtime grows with the number of trials and five-fold fits
OPTUNA_TRIALS_RV = 200
OPTUNA_TRIALS_CSW = 200

# Start with five random trials before using earlier results to guide the search
OPTUNA_STARTUP_TRIALS = 5

# Keep the RV cap; allow more CSW trees because earlier runs approached the cap
RV_MAX_ESTIMATORS = 3000
CSW_MAX_ESTIMATORS = 6000
# Stop adding trees after this many rounds without validation improvement
EARLY_STOPPING_ROUNDS = 150

# Fixed settings outside the search
COMMON_LGB_PARAMS = {
    "subsample_freq": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# Test the original V2 settings first in a new study
ORIGINAL_SEARCH_POINT = {
    "learning_rate": 0.01,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Print progress and warning messages and stop the program for fatal errors
def info(msg): print(f"[info] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)
def fail(msg): print(f"[FATAL] {msg}", flush=True); sys.exit(1)


# Convert right to +1, left to -1, and unknown handedness to 0
def hand_sign(series):
    s = series.astype(str).str.strip().str.lower()
    return np.where(s.isin(["right", "r"]), 1.0,
                    np.where(s.isin(["left", "l"]), -1.0, 0.0))


# Replace special characters so pitch types can be used as filenames
def safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_") or "unknown"


# Subtract the angle expected from release height using a straight-line fit
def height_adjusted_vaa(df):
    if VAA_COL not in df.columns or REL_HEIGHT_COL not in df.columns:
        return None
    v = pd.to_numeric(df[VAA_COL], errors="coerce")
    h = pd.to_numeric(df[REL_HEIGHT_COL], errors="coerce")
    mask = v.notna() & h.notna()
    if mask.sum() < 100:
        return None
    # Learn this adjustment from all prepared rows before cross-validation
    slope, intercept = np.polyfit(h[mask].values, v[mask].values, 1)
    resid = v - (slope * h + intercept)
    info(f"Height-adjusted VAA: slope={slope:.4f} intercept={intercept:.4f}")
    return resid


# Load the pitch data and create the targets and model inputs
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

    # Keep missing or blank PitchCall labels missing, rather than treating them as CSW=0
    ev = df[EVENT_SOURCE_COL].astype("string").str.strip()
    valid_csw_label = ev.notna() & ev.ne("")
    if not any(e in set(ev.dropna().unique()) for e in CSW_EVENTS):
        warn(f"None of CSW_EVENTS {CSW_EVENTS} found. Seen: {sorted(set(ev.dropna().unique()))[:15]}")
    df["csw"] = np.where(valid_csw_label, ev.isin(CSW_EVENTS).astype(int), np.nan)

    # Filter RV and CSW independently in main()

    # Convert available measurements to numbers and record their input names
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

    # Mirror horizontal measurements so left- and right-handers share a reference direction
    if PITCHER_HAND_COL in df.columns:
        psign = hand_sign(df[PITCHER_HAND_COL])
        for f in HAND_SIGNED_FEATURES:
            if f in df.columns:
                df[f] = df[f] * psign
    else:
        warn(f"PITCHER_HAND_COL '{PITCHER_HAND_COL}' missing.")

    new_cols = {}

    # Add the vertical approach angle after adjusting for release height
    resid = height_adjusted_vaa(df)
    if resid is not None:
        new_cols["VertApprAngle_hadj"] = resid.values
        feature_names.append("VertApprAngle_hadj")

    # Calculate each pitch’s difference from its pitcher and pitch-type average
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

    # Add the measurement products listed in the settings
    for a, b in INTERACTION_PAIRS:
        if a in df.columns and b in df.columns:
            name = f"{a}_x_{b}"
            new_cols[name] = (df[a] * df[b]).values
            feature_names.append(name)
        else:
            warn(f"Interaction ({a}, {b}) skipped -- a base feature is missing.")

    # Mark right-handed batters as 1 and all other values as 0
    if BATTER_HAND_COL in df.columns:
        new_cols["batter_is_rhh"] = (hand_sign(df[BATTER_HAND_COL]) > 0).astype(int)
        feature_names.append("batter_is_rhh")
    else:
        warn(f"BATTER_HAND_COL '{BATTER_HAND_COL}' missing.")

    # Attach the new inputs together to avoid repeatedly expanding the table
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()

    # Remove pitches without release speed
    if "RelSpeed" in df.columns:
        df = df.dropna(subset=["RelSpeed"]).reset_index(drop=True)

    info(f"Final feature set ({len(feature_names)}): {feature_names}")
    info(f"Overall CSW rate among valid PitchCall rows: {df['csw'].mean():.3%}")
    return df, feature_names


# Keep every pitch from a given pitcher in the same validation fold
def get_folds(df, target_col):
    """[CHANGED VS V2] Build pitcher-grouped folds for the target-specific dataset."""
    if PITCHER_ID_COL in df.columns:
        groups = df[PITCHER_ID_COL].astype(str).values
    else:
        warn(f"PITCHER_ID_COL '{PITCHER_ID_COL}' missing -- pseudo groups.")
        # Fall back to artificial groups when pitcher names are missing
        groups = (np.arange(len(df)) % (N_SPLITS * 50)).astype(str)
    return list(GroupKFold(n_splits=N_SPLITS).split(df, df[target_col], groups))


# Check for empty folds, overlapping pitchers, and missing CSW classes before tuning
def validate_folds_before_tuning(df, folds, target_col, is_clf, label):
    """[OVERNIGHT-SAFE UPDATE] Fail fast before a long run if grouped CV is malformed."""
    if len(folds) != N_SPLITS:
        fail(f"{label}: expected {N_SPLITS} folds but received {len(folds)}.")

    if PITCHER_ID_COL in df.columns:
        groups = df[PITCHER_ID_COL].astype(str).reset_index(drop=True)
    else:
        groups = None

    y = df[target_col].reset_index(drop=True)
    for fold_num, (tr, va) in enumerate(folds, start=1):
        if len(tr) == 0 or len(va) == 0:
            fail(f"{label}: fold {fold_num} has an empty train/validation split.")
        if groups is not None:
            overlap = set(groups.iloc[tr]) & set(groups.iloc[va])
            if overlap:
                fail(
                    f"{label}: pitcher leakage in fold {fold_num}; "
                    f"example={next(iter(overlap))}"
                )
        if is_clf:
            if y.iloc[tr].nunique() < 2:
                fail(f"{label}: fold {fold_num} training data has only one CSW class.")
            if y.iloc[va].nunique() < 2:
                warn(
                    f"{label}: fold {fold_num} validation data has only one CSW class; "
                    "AUC would be undefined for that fold."
                )

    info(
        f"{label}: preflight passed ({len(df):,} rows, "
        f"{df[PITCHER_ID_COL].nunique() if PITCHER_ID_COL in df.columns else 'pseudo'} pitcher groups)."
    )


# Choose one trial’s settings from the allowed search ranges
def suggest_lgb_params(trial, is_clf):
    """[ADDED VS V2] Define the Optuna search space for one LightGBM target."""
    max_estimators = CSW_MAX_ESTIMATORS if is_clf else RV_MAX_ESTIMATORS
    params = {
        # Learning rate controls how much each new tree changes predictions
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        # More leaves allow each tree to learn more detailed patterns
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        # Require a minimum number of pitches in each leaf
        "min_child_samples": trial.suggest_int("min_child_samples", 25, 400, log=True),
        # Choose the fraction of training pitches sampled for a tree
        "subsample": trial.suggest_float("subsample", 0.70, 1.00),
        # Choose the fraction of inputs available to each tree
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
        # Penalize leaf values to limit overly complex predictions
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        # Apply a second penalty that also limits large leaf values
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "n_estimators": max_estimators,
        **COMMON_LGB_PARAMS,
    }
    return params


# Test settings on the same pitcher folds and save the best result
def tune_hyperparameters(df, feature_names, folds, target_col, is_clf, label, n_trials):
    """
    [ADDED VS V2] Tune LightGBM with pitcher-grouped cross-validation.

    RV minimizes mean fold RMSE. CSW maximizes mean fold AUC. Early stopping is
    applied inside every fold, so n_estimators is only a ceiling. The final model
    still uses the median best iteration from CV, preserving Version 2's refit logic.
    """
    X = df[feature_names]
    y = df[target_col].astype(int if is_clf else float)
    Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
    obj = "binary" if is_clf else "regression"
    eval_metric = "auc" if is_clf else "rmse"

    print(f"\n{'=' * 70}")
    print(f"Optuna tuning: {label} ({n_trials} trials)")
    print(f"{'=' * 70}")

    # Return average validation RMSE for RV or AUC for CSW for one trial
    def objective(trial):
        params = suggest_lgb_params(trial, is_clf)
        primary_scores = []
        secondary_scores = []
        best_iters = []

        for fold_num, (tr, va) in enumerate(folds, start=1):
            model = Est(objective=obj, **params)
            model.fit(
                X.iloc[tr], y.iloc[tr],
                eval_set=[(X.iloc[va], y.iloc[va])],
                eval_metric=eval_metric,
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            best_iters.append(int(model.best_iteration_ or params["n_estimators"]))

            if is_clf:
                pred = model.predict_proba(X.iloc[va])[:, 1]
                # Optimize CSW AUC; retain log loss for diagnostics
                if y.iloc[va].nunique() > 1:
                    primary_scores.append(roc_auc_score(y.iloc[va], pred))
                secondary_scores.append(log_loss(y.iloc[va], pred, labels=[0, 1]))
            else:
                pred = model.predict(X.iloc[va])
                primary_scores.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
                secondary_scores.append(mean_absolute_error(y.iloc[va], pred))

        # Use chance-level AUC if no fold has both CSW outcomes
        if is_clf and not primary_scores:
            return 0.5

        # Save the tree count and secondary score alongside the trial
        trial.set_user_attr("median_best_iteration", int(np.median(best_iters)))
        trial.set_user_attr("secondary_metric_mean", float(np.mean(secondary_scores)))
        return float(np.mean(primary_scores))

    direction = "maximize" if is_clf else "minimize"

    # Persist trials in SQLite so interrupted searches can resume
    sub = os.path.join(OUT_DIR, target_col)
    os.makedirs(sub, exist_ok=True)
    db_path = os.path.abspath(os.path.join(sub, f"optuna_{target_col}.db"))
    storage_url = f"sqlite:///{db_path}"
    study_name = f"v2_optuna_{target_col}"

    # Open the saved search or create one using results to guide later trials
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE,
            n_startup_trials=OPTUNA_STARTUP_TRIALS,
        ),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    # Queue the original settings only for a new study
    if len(study.trials) == 0:
        study.enqueue_trial(ORIGINAL_SEARCH_POINT)

    # Count completed or pruned trials toward the budget and replace failed trials
    finished_states = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    finished = sum(t.state in finished_states for t in study.trials)
    remaining = max(0, n_trials - finished)
    info(
        f"{label}: persistent Optuna study has {finished} finished trial(s); "
        f"running {remaining} more to reach {n_trials}."
    )
    info(f"{label}: study database -> {db_path}")

    checkpoint_csv = os.path.join(sub, f"optuna_trials_{target_col}.csv")

    def checkpoint_callback(study_obj, trial_obj):
        # Export trial history after every trial
        study_obj.trials_dataframe().to_csv(checkpoint_csv, index=False)
        state = trial_obj.state.name
        try:
            best_value = study_obj.best_value
            metric_name = "AUC" if is_clf else "RMSE"
            info(
                f"{label}: trial {trial_obj.number} finished ({state}); "
                f"current best {metric_name}={best_value:.6f}"
            )
        except ValueError:
            info(f"{label}: trial {trial_obj.number} finished ({state}); no complete trial yet.")

    # Run only the trials still needed to reach the requested total
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=False,
            callbacks=[checkpoint_callback],
            # Continue after LightGBM failures; let unexpected errors stop the run
            catch=(lgb.basic.LightGBMError,),
            gc_after_trial=True,
        )

    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        fail(f"{label}: Optuna finished without any successful trials.")

    # Combine the winning search values with settings that stayed fixed
    best = study.best_trial
    best_params = dict(best.params)
    best_params.update(COMMON_LGB_PARAMS)
    best_params["n_estimators"] = CSW_MAX_ESTIMATORS if is_clf else RV_MAX_ESTIMATORS

    # Save final trial history and the winning configuration
    study.trials_dataframe().to_csv(
        os.path.join(sub, f"optuna_trials_{target_col}.csv"), index=False
    )
    with open(os.path.join(sub, f"best_params_{target_col}.json"), "w") as f:
        json.dump({
            "best_value": float(best.value),
            "best_params": best_params,
            "median_best_iteration_during_tuning": best.user_attrs.get("median_best_iteration"),
            "secondary_metric_mean": best.user_attrs.get("secondary_metric_mean"),
            "n_trials": n_trials,
            "direction": direction,
        }, f, indent=2)

    metric_name = "AUC" if is_clf else "RMSE"
    info(f"{label}: Optuna best {metric_name}={best.value:.6f}")
    info(f"{label}: Optuna best params={best_params}")
    return best_params



def median_best_iter(X, y, folds, is_clf, params, label="Model"):
    # Predict run value as a number and CSW as a probability
    Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
    obj = "binary" if is_clf else "regression"
    metric = "auc" if is_clf else "rmse"

    # Collect tree counts and validation metrics by fold
    iters = []
    fold_results = []

    # Print the CV report header
    print(f"\n{'=' * 60}")
    print(f"Cross-validation performance: {label}")
    print(f"{'=' * 60}")

    for fold_num, (tr, va) in enumerate(folds, start=1):
        # Keep pitchers separate between training and validation and stop when progress stalls
        m = Est(objective=obj, **params)

        m.fit(
            X.iloc[tr],
            y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            eval_metric=metric,
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0)
            ],
        )

        # Save the best tree count from each fold for the final training step
        best_iter = int(
            m.best_iteration_ or params["n_estimators"]
        )
        iters.append(best_iter)

        # Report AUC/log loss for CSW and RMSE/MAE for RV
        if is_clf:
            pred = m.predict_proba(X.iloc[va])[:, 1]

            auc = roc_auc_score(y.iloc[va], pred) if y.iloc[va].nunique() > 1 else np.nan
            ll = log_loss(y.iloc[va], pred, labels=[0, 1])

            fold_results.append({
                "fold": fold_num,
                "best_iteration": best_iter,
                "auc": auc,
                "log_loss": ll,
            })

            print(
                f"Fold {fold_num}: "
                f"AUC={auc:.4f} | "
                f"Log Loss={ll:.4f} | "
                f"Best Iter={best_iter}"
            )

        else:
            pred = m.predict(X.iloc[va])

            rmse = np.sqrt(
                mean_squared_error(y.iloc[va], pred)
            )
            mae = mean_absolute_error(y.iloc[va], pred)

            fold_results.append({
                "fold": fold_num,
                "best_iteration": best_iter,
                "rmse": rmse,
                "mae": mae,
            })

            print(
                f"Fold {fold_num}: "
                f"RMSE={rmse:.6f} | "
                f"MAE={mae:.6f} | "
                f"Best Iter={best_iter}"
            )

    # Collect fold metrics for reporting and export
    results_df = pd.DataFrame(fold_results)

    print("\nAverage CV performance:")

    if is_clf:
        print(
            f"AUC:      {results_df['auc'].mean():.4f} "
            f"+/- {results_df['auc'].std():.4f}"
        )
        print(
            f"Log Loss: {results_df['log_loss'].mean():.4f} "
            f"+/- {results_df['log_loss'].std():.4f}"
        )
    else:
        print(
            f"RMSE: {results_df['rmse'].mean():.6f} "
            f"+/- {results_df['rmse'].std():.6f}"
        )
        print(
            f"MAE:  {results_df['mae'].mean():.6f} "
            f"+/- {results_df['mae'].std():.6f}"
        )

    # Use the middle tree count across folds so one unusual fold has less influence
    median_iter = int(np.median(iters))

    print(f"Median best iteration: {median_iter}")
    print("=" * 60)

    return median_iter, results_df


# Train on all eligible pitches using the selected number of trees
def train_full(X, y, n_estimators, is_clf, params):
    Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
    obj = "binary" if is_clf else "regression"
    p = dict(params)
    p["n_estimators"] = n_estimators
    m = Est(objective=obj, **p)
    m.fit(X, y)
    return m


# Turn predictions into grades centered at 100 with higher always meaning better
def scale_grade(pred, invert):
    v = -np.asarray(pred) if invert else np.asarray(pred)
    # Divide by the prediction spread and protect against division by zero
    z = (v - v.mean()) / (v.std(ddof=0) + 1e-12)
    return SCALE_MEAN + SCALE_SD * z


# Plot how much each input helped reduce training error through tree splits
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


# Summarize each input’s prediction contributions separately for each pitch type
def save_per_pitchtype_importance(model, X, pitch_types, feature_names, target_col, label):
    if pitch_types is None:
        warn(f"{label}: no pitch-type column -- skipping per-pitch-type importance.")
        return
    # Get SHAP contributions and drop the final column containing the base prediction
    contribs = np.asarray(model.booster_.predict(X, pred_contrib=True))[:, :len(feature_names)]
    abs_df = pd.DataFrame(np.abs(contribs), columns=feature_names)
    abs_df["_pt"] = np.asarray(pitch_types)

    counts = abs_df["_pt"].value_counts()
    keep_types = counts[counts >= MIN_PITCHES_PER_GROUP].index.tolist()
    abs_df = abs_df[abs_df["_pt"].isin(keep_types)]
    if not keep_types:
        warn(f"{label}: no pitch types clear the sample threshold -- skipping heatmap.")
        return

    mat = abs_df.groupby("_pt").mean().T
    order = mat.mean(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order]
    # Express each input’s average absolute contribution as a share within its pitch type
    norm = mat / (mat.sum(axis=0) + 1e-12)

    sub = os.path.join(OUT_DIR, target_col)
    mat.to_csv(os.path.join(sub, f"pitchtype_importance_raw_{target_col}.csv"))
    norm.to_csv(os.path.join(sub, f"pitchtype_importance_share_{target_col}.csv"))

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


# Check the winning settings then train, grade, and save one target’s outputs
def score_target(df, feature_names, folds, target_col, is_clf, invert, label, params):
    info(f"=== {label} ===")
    X = df[feature_names]
    y = df[target_col].astype(int if is_clf else float)
    n, results_df = median_best_iter(X, y, folds, is_clf, params, label=label)
    info(f"{label}: median best_iter={n} (cap {params['n_estimators']}); refitting on full data ...")
    if n >= params["n_estimators"] - 1:
        warn(f"{label}: best_iter hit the n_estimators cap -- consider raising it further.")
    m = train_full(X, y, n, is_clf, params)
    # These final grades score the same pitches used to train the full model
    pred = m.predict_proba(X)[:, 1] if is_clf else m.predict(X)
    grade = scale_grade(pred, invert)
    sub = os.path.join(OUT_DIR, target_col)
    os.makedirs(sub, exist_ok=True)
    results_df.to_csv(os.path.join(sub, f"cv_performance_{target_col}.csv"), index=False)
    m.booster_.save_model(os.path.join(sub, f"lightgbm_{target_col}.txt"))
    save_importance_chart(m, feature_names, target_col, label)
    pt = df[PITCH_TYPE_COL] if PITCH_TYPE_COL in df.columns else None
    save_per_pitchtype_importance(m, X, pt, feature_names, target_col, label)
    return grade


# Save Stuff+ and CSW+ rankings within each pitch type
def build_pitchtype_leaderboards(grp):
    bt_dir = os.path.join(OUT_DIR, "leaderboards_by_pitchtype")
    os.makedirs(bt_dir, exist_ok=True)
    cols = [PITCHER_ID_COL, "pitches", "rv_pitches", "csw_pitches", "stuff_plus", "csw_plus"]

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
        print(g_stuff.head(TOP_N_PRINT)[[PITCHER_ID_COL, "pitches", "rv_pitches", "csw_pitches", "stuff_plus",
                                         "csw_plus"]].to_string(index=False), flush=True)
        print(f"Top {TOP_N_PRINT} '{pt}' by CSW+:", flush=True)
        print(g.sort_values("csw_plus", ascending=False).head(TOP_N_PRINT)[
            [PITCHER_ID_COL, "pitches", "rv_pitches", "csw_pitches", "stuff_plus", "csw_plus"]].to_string(index=False),
            flush=True)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_path = os.path.join(OUT_DIR, "leaderboards_by_pitchtype_all.csv")
        combined.to_csv(combined_path, index=False)
        info(f"Saved per-pitch-type leaderboards -> {bt_dir}\\ and {combined_path}")


# Run data preparation, tuning, final training, and report generation in order
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 74, flush=True)
    print(" Version 2 + Optuna tuning + independent RV/CSW target filtering", flush=True)
    print("=" * 74, flush=True)

    df, feature_names = load_and_prepare()

    # Keep each target’s valid rows independently
    df_rv = df.dropna(subset=["run_value"]).copy()
    df_csw = df.dropna(subset=["csw"]).copy()
    info(f"RV eligible pitches:  {len(df_rv):,}")
    info(f"CSW eligible pitches: {len(df_csw):,}")

    # Build pitcher-grouped folds separately for each target
    rv_folds = get_folds(df_rv, "run_value")
    csw_folds = get_folds(df_csw, "csw")

    # Validate folds before the tuning run
    validate_folds_before_tuning(
        df_rv, rv_folds, "run_value", is_clf=False, label="Stuff+ (run value)"
    )
    validate_folds_before_tuning(
        df_csw, csw_folds, "csw", is_clf=True, label="CSW+"
    )

    # Tune each target before CV reporting and full-data refitting; CSW has a higher cap
    rv_params = tune_hyperparameters(
        df_rv, feature_names, rv_folds, "run_value",
        is_clf=False, label="Stuff+ (run value)", n_trials=OPTUNA_TRIALS_RV,
    )
    csw_params = tune_hyperparameters(
        df_csw, feature_names, csw_folds, "csw",
        is_clf=True, label="CSW+", n_trials=OPTUNA_TRIALS_CSW,
    )

    stuff_plus = score_target(
        df_rv, feature_names, rv_folds, "run_value",
        is_clf=False, invert=True, label="Stuff+ (run value)", params=rv_params,
    )
    csw_plus = score_target(
        df_csw, feature_names, csw_folds, "csw",
        is_clf=True, invert=False, label="CSW+", params=csw_params,
    )

    # Match grades back to the original rows even when only one target is available
    scored = df[[c for c in [PITCHER_ID_COL, PITCH_TYPE_COL] if c in df.columns]].copy()
    scored["run_value"] = df["run_value"].values
    scored["csw"] = df["csw"].values
    scored["stuff_plus"] = np.nan
    scored["csw_plus"] = np.nan
    scored.loc[df_rv.index, "stuff_plus"] = stuff_plus
    scored.loc[df_csw.index, "csw_plus"] = csw_plus
    scored.to_csv(os.path.join(OUT_DIR, "scored_pitches.csv"), index=False)
    info(f"Saved per-pitch scores -> {os.path.join(OUT_DIR, 'scored_pitches.csv')}")

    if PITCHER_ID_COL in df.columns and PITCH_TYPE_COL in df.columns:
        # Track separate RV/CSW counts because eligible rows may differ
        grp = (scored.groupby([PITCHER_ID_COL, PITCH_TYPE_COL])
                     .agg(pitches=("run_value", "size"),
                          rv_pitches=("stuff_plus", "count"),
                          csw_pitches=("csw_plus", "count"),
                          stuff_plus=("stuff_plus", "mean"),
                          csw_plus=("csw_plus", "mean")).reset_index())

        # Apply the 50-pitch minimum independently to each target
        grp.loc[grp["rv_pitches"] < MIN_PITCHES_PER_GROUP, "stuff_plus"] = np.nan
        grp.loc[grp["csw_pitches"] < MIN_PITCHES_PER_GROUP, "csw_plus"] = np.nan
        grp = grp[grp["stuff_plus"].notna() | grp["csw_plus"].notna()].copy()
        grp["stuff_plus"] = grp["stuff_plus"].round(1)
        grp["csw_plus"] = grp["csw_plus"].round(1)
        grp = grp.sort_values([PITCHER_ID_COL, "stuff_plus"], ascending=[True, False])
        path = os.path.join(OUT_DIR, "pitcher_pitch_grades.csv")
        grp.to_csv(path, index=False)
        info(f"Saved per pitcher-pitch-type grades ({len(grp)} rows) -> {path}")

        # Weight each available pitch-type grade by its valid pitch count
        def weighted(g, col, weight_col):
            valid = g[col].notna() & g[weight_col].gt(0)
            if not valid.any():
                return np.nan
            return float(np.average(g.loc[valid, col].values,
                                    weights=g.loc[valid, weight_col].values))

        # Combine qualifying pitch types into one overall row for each pitcher
        lb = (grp.groupby(PITCHER_ID_COL)
                 .apply(lambda g: pd.Series({
                     "pitches": int(g["pitches"].sum()),
                     "rv_pitches": int(g["rv_pitches"].sum()),
                     "csw_pitches": int(g["csw_pitches"].sum()),
                     "pitch_types": int(g[PITCH_TYPE_COL].nunique()),
                     "stuff_plus": weighted(g, "stuff_plus", "rv_pitches"),
                     "csw_plus": weighted(g, "csw_plus", "csw_pitches"),
                 }))
                 .reset_index())
        lb = lb[lb["pitches"] >= MIN_PITCHES_PER_PITCHER].copy()
        lb["pitches"] = lb["pitches"].astype(int)
        lb["rv_pitches"] = lb["rv_pitches"].astype(int)
        lb["csw_pitches"] = lb["csw_pitches"].astype(int)
        lb["pitch_types"] = lb["pitch_types"].astype(int)
        lb["stuff_plus"] = lb["stuff_plus"].round(1)
        lb["csw_plus"] = lb["csw_plus"].round(1)
        lb = lb.sort_values("stuff_plus", ascending=False)
        lb_path = os.path.join(OUT_DIR, "pitcher_leaderboard.csv")
        lb.to_csv(lb_path, index=False)
        info(f"Saved overall pitcher leaderboard ({len(lb)} pitchers) -> {lb_path}")

        print("\nTop 25 pitchers by arsenal-weighted Stuff+ (min "
              f"{MIN_PITCHES_PER_PITCHER} pitches):", flush=True)
        print(lb.head(25).to_string(index=False), flush=True)
        print("\nTop 25 pitchers by arsenal-weighted CSW+:", flush=True)
        print(lb.sort_values("csw_plus", ascending=False).head(25).to_string(index=False),
              flush=True)

        print(f"\nTop 25 pitcher pitch-types by Stuff+ (min "
              f"{MIN_PITCHES_PER_GROUP} valid RV pitches):", flush=True)
        print(grp.sort_values("stuff_plus", ascending=False).head(25).to_string(index=False),
              flush=True)
        print(f"\nTop 25 pitcher pitch-types by CSW+ (min "
              f"{MIN_PITCHES_PER_GROUP} valid CSW pitches):", flush=True)
        print(grp.sort_values("csw_plus", ascending=False).head(25).to_string(index=False),
              flush=True)

        build_pitchtype_leaderboards(grp)
    else:
        warn("Cannot build per pitcher-pitch-type grades (missing pitcher or pitch-type column).")

    # Save the settings and row counts needed to understand this run
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump({
            "n_pitches_after_common_preprocessing": int(len(df)),
            "n_rv_pitches": int(len(df_rv)),
            "n_csw_pitches": int(len(df_csw)),
            "features": feature_names,
            "min_pitches_per_group": MIN_PITCHES_PER_GROUP,
            "min_pitches_per_pitcher": MIN_PITCHES_PER_PITCHER,
            "interaction_pairs": [f"{a}_x_{b}" for a, b in INTERACTION_PAIRS],
            "target_filtering": "RV and CSW filtered independently",
            "missing_pitchcall": "missing PitchCall remains missing CSW target",
            "tuning": "Optuna with pitcher-grouped 5-fold cross-validation",
            "optuna_trials_rv": OPTUNA_TRIALS_RV,
            "optuna_trials_csw": OPTUNA_TRIALS_CSW,
            "rv_max_estimators": RV_MAX_ESTIMATORS,
            "csw_max_estimators": CSW_MAX_ESTIMATORS,
            "rv_best_params": rv_params,
            "csw_best_params": csw_params,
        }, f, indent=2)

    print("\nDone. See the '%s' folder." % OUT_DIR, flush=True)


# Start training only when this file is run directly
if __name__ == "__main__":
    main()