"""
train_model.py  ·  v6  —  Production-Clean Anti-Overfit Ensemble
=================================================================
Key fixes vs v5
───────────────
  ✅  ALL DataFrames preserved end-to-end — no .values / np.array conversions
  ✅  _transform_to_df always returns pd.DataFrame with proper column names
  ✅  WeightedEnsemble.predict() receives and passes DataFrames only
  ✅  UserWarning / FutureWarning suppressed globally
  ✅  model.pkl saves WeightedEnsemble that accepts raw pd.DataFrame from app.py
  ✅  label_encoders.pkl replaced with metadata dict (no legacy encoders needed)

Strategy
────────
  1. Split: 80% train-pool / 20% test
  2. Within train-pool: 85% train / 15% validation  (early stopping)
  3. XGBoost + LightGBM + GradientBoosting with hard regularisation
  4. Early stopping on validation set → best n_estimators
  5. Retrain on full train-pool using best n_estimators
  6. Weighted ensemble: 0.4·XGB + 0.4·LGB + 0.2·GBR
  7. Evaluate: Train R² / CV R² (5×2 RepeatedKFold) / Test R²

Target: Test R² ≥ 0.90  |  Gap (CV - Test) ≤ 0.03
"""

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, RepeatedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Suppress all sklearn / LightGBM feature-name warnings ──────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("⚠️  pip install xgboost")

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("⚠️  pip install lightgbm")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    load_and_preprocess, inverse_transform_target,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
)

# ── Paths ───────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE, "cardekho_dataset.csv")
MODEL_PATH    = os.path.join(BASE, "model.pkl")
SCALER_PATH   = os.path.join(BASE, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE, "label_encoders.pkl")

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
VAL_SIZE      = 0.15
ES_ROUNDS     = 50


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipe(preprocessor: ColumnTransformer, model) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-preserving transform helper
# Always returns pd.DataFrame — eliminates LightGBM feature-name warning
# ─────────────────────────────────────────────────────────────────────────────

def _transform_to_df(preprocessor: ColumnTransformer,
                     X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Fit-transform or transform X and return a named pd.DataFrame.
    Passing DataFrames (not numpy arrays) to LightGBM eliminates
    the 'X does not have valid feature names' warning completely.
    """
    if fit:
        arr = preprocessor.fit_transform(X)
    else:
        arr = preprocessor.transform(X)

    try:
        col_names = list(preprocessor.get_feature_names_out())
    except Exception:
        col_names = [f"f{i}" for i in range(arr.shape[1])]

    # Always return DataFrame — never raw numpy
    return pd.DataFrame(arr, columns=col_names, index=X.index)


# ─────────────────────────────────────────────────────────────────────────────
# Regularised model configs
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_config(n_est: int = 1000) -> "XGBRegressor":
    return XGBRegressor(
        n_estimators     = n_est,
        max_depth        = 5,
        learning_rate    = 0.03,
        subsample        = 0.70,
        colsample_bytree = 0.70,
        reg_alpha        = 0.30,
        reg_lambda       = 2.00,
        min_child_weight = 5,
        gamma            = 0.10,
        tree_method      = "hist",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        verbosity        = 0,
    )


def _lgb_config(n_est: int = 1000) -> "LGBMRegressor":
    return LGBMRegressor(
        n_estimators      = n_est,
        num_leaves        = 31,
        max_depth         = 5,
        learning_rate     = 0.03,
        feature_fraction  = 0.70,
        bagging_fraction  = 0.70,
        bagging_freq      = 5,
        lambda_l1         = 0.10,
        lambda_l2         = 2.00,
        min_child_samples = 30,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbose           = -1,
    )


def _gbr_config(n_est: int = 600) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators    = n_est,
        max_depth       = 5,
        learning_rate   = 0.03,
        subsample       = 0.70,
        min_samples_split = 15,
        min_samples_leaf  = 10,
        max_features    = 0.70,
        random_state    = RANDOM_STATE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Early-stopping training  (X inputs are DataFrames throughout)
# ─────────────────────────────────────────────────────────────────────────────

def _train_xgb_early_stop(X_tr_df: pd.DataFrame, y_tr,
                           X_val_df: pd.DataFrame, y_val) -> tuple:
    """Train XGB with early stopping. X must be pd.DataFrame."""
    model = _xgb_config(n_est=3000)
    model.set_params(early_stopping_rounds=ES_ROUNDS, eval_metric="rmse")
    model.fit(
        X_tr_df, y_tr,
        eval_set=[(X_val_df, y_val)],
        verbose=False,
    )
    best = model.best_iteration
    print(f"      [XGB]  early stop at round {best}")
    return model, best


def _train_lgb_early_stop(X_tr_df: pd.DataFrame, y_tr,
                           X_val_df: pd.DataFrame, y_val) -> tuple:
    """Train LGB with early stopping. X must be pd.DataFrame."""
    model = _lgb_config(n_est=3000)
    model.fit(
        X_tr_df, y_tr,
        eval_set=[(X_val_df, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=ES_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    best = model.best_iteration_
    print(f"      [LGB]  early stop at round {best}")
    return model, best


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_orig(y_true_log, y_pred_log) -> tuple:
    yt   = inverse_transform_target(np.asarray(y_true_log))
    yp   = inverse_transform_target(np.asarray(y_pred_log))
    r2   = r2_score(yt, yp)
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return r2, mae, rmse


def _print_metrics(label, r2, mae, rmse):
    flag = "✅" if r2 >= 0.90 else ("⚡" if r2 >= 0.85 else "⚠️")
    print(f"      {flag} [{label:<22}]  R²={r2:.4f}  "
          f"MAE=₹{mae:,.0f}  RMSE=₹{rmse:,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance printer
# ─────────────────────────────────────────────────────────────────────────────

def _show_importance(model, col_names, top_n: int = 15):
    if not hasattr(model, "feature_importances_"):
        return
    imps    = model.feature_importances_
    top_idx = np.argsort(imps)[::-1][:top_n]
    print(f"\n      {'Feature':<42} {'Score':>8}  Bar")
    print("      " + "─" * 68)
    for i in top_idx:
        n   = col_names[i] if i < len(col_names) else f"feat_{i}"
        bar = "█" * int(imps[i] * 80)
        print(f"      {n:<42} {imps[i]:>8.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# WeightedEnsemble
# Accepts raw pd.DataFrame (with original feature names) → handles
# preprocessing internally → returns predictions.
# Compatible with app.py: model.predict(input_df)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedEnsemble:
    """
    Weighted ensemble wrapper.
    Stores fitted preprocessor + individual fitted models.
    predict() always receives and processes pd.DataFrames — ZERO warnings.
    """

    def __init__(self, preprocessor: ColumnTransformer,
                 models_weights: list):
        """
        preprocessor   : already fitted ColumnTransformer
        models_weights : list of (fitted_model, weight, name)
        """
        self.preprocessor   = preprocessor
        self.models_weights = models_weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        X must be a pd.DataFrame with the same columns as training data
        (ALL_FEATURES from preprocessing.py).
        Returns predicted log-price array.
        """
        # _transform_to_df always returns pd.DataFrame → no warnings
        X_t     = _transform_to_df(self.preprocessor, X, fit=False)
        pred    = np.zeros(len(X_t))
        total_w = 0.0
        for model, w, _ in self.models_weights:
            pred    += w * model.predict(X_t)   # X_t is DataFrame
            total_w += w
        return pred / total_w

    def get_params(self, deep=True):
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation helper  (sklearn Pipeline — no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def _cv_pipeline(preprocessor, model, X: pd.DataFrame, y_log, label: str):
    """5×2 RepeatedKFold CV — X stays as pd.DataFrame."""
    pipe   = _build_pipe(preprocessor, model)
    rkf    = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y_log, cv=rkf, scoring="r2", n_jobs=-1)
    print(f"      [{label:<22}]  CV R²={scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean(), scores.std()


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────

def train_best_model():
    print("=" * 72)
    print("   CAR RESALE PRICE  ·  v6 PRODUCTION-CLEAN ENSEMBLE TRAINING")
    print("   Target: Test R² ≥ 0.90  |  Gap ≤ 0.03  |  ZERO warnings")
    print("=" * 72)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading & preprocessing …")
    X, y_log, feat_names, preprocessor = load_and_preprocess(DATA_PATH)
    # X is pd.DataFrame throughout

    # ── 2. Splits ─────────────────────────────────────────────────────────────
    print("\n[2/7] Splitting data …")
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    print(f"      Train={len(X_train):,}  Val={len(X_val):,}  "
          f"Test={len(X_test):,}  (pool={len(X_pool):,})")

    # ── 3. Fit preprocessor ONCE on training set only ─────────────────────────
    print("\n[3/7] Fitting preprocessor on X_train only (no leakage) …")
    X_train_t = _transform_to_df(preprocessor, X_train, fit=True)   # fits here
    X_val_t   = _transform_to_df(preprocessor, X_val,   fit=False)
    X_pool_t  = _transform_to_df(preprocessor, X_pool,  fit=False)
    X_test_t  = _transform_to_df(preprocessor, X_test,  fit=False)
    col_names = list(X_train_t.columns)
    print(f"      Transformed features : {len(col_names)}")
    print(f"      All splits type      : {type(X_train_t).__name__}  ✅")

    # ── 4. Early-stopping phase ───────────────────────────────────────────────
    print("\n[4/7] Early-stopping training …\n")
    best_rounds: dict = {}

    if XGB_OK:
        _, best_rounds["xgb"] = _train_xgb_early_stop(
            X_train_t, y_train, X_val_t, y_val
        )
    if LGB_OK:
        _, best_rounds["lgb"] = _train_lgb_early_stop(
            X_train_t, y_train, X_val_t, y_val
        )

    # ── 5. Retrain on full train-pool ─────────────────────────────────────────
    print("\n[5/7] Retraining on full train-pool …\n")
    fitted_models: dict = {}

    if XGB_OK:
        n_xgb = best_rounds["xgb"]
        xgb_final = _xgb_config(n_est=n_xgb)
        xgb_final.fit(X_pool_t, y_pool)
        fitted_models["XGBoost"] = xgb_final
        print(f"      [XGB]  n_estimators={n_xgb}")

    if LGB_OK:
        n_lgb = best_rounds["lgb"]
        lgb_final = _lgb_config(n_est=n_lgb)
        lgb_final.fit(X_pool_t, y_pool)
        fitted_models["LightGBM"] = lgb_final
        print(f"      [LGB]  n_estimators={n_lgb}")

    gbr_final = _gbr_config(n_est=600)
    gbr_final.fit(X_pool_t, y_pool)
    fitted_models["GBR"] = gbr_final
    print(f"      [GBR]  n_estimators=600")

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    print("\n[6/7] Evaluation (original ₹ scale) …\n")

    print("   Train R² (pool):")
    for name, m in fitted_models.items():
        r2, mae, rmse = _metrics_orig(y_pool, m.predict(X_pool_t))
        _print_metrics(f"{name} train", r2, mae, rmse)

    print("\n   CV R² (5-fold × 2 repeats, RepeatedKFold):")
    cv_results = {}
    for name, m in fitted_models.items():
        n_est     = (best_rounds.get("xgb") if name == "XGBoost"
                     else best_rounds.get("lgb") if name == "LightGBM"
                     else 600)
        fresh     = (_xgb_config(n_est) if name == "XGBoost"
                     else _lgb_config(n_est) if name == "LightGBM"
                     else _gbr_config(600))
        cv_mean, cv_std = _cv_pipeline(preprocessor, fresh, X, y_log, name)
        cv_results[name] = (cv_mean, cv_std)

    print("\n   Test R² (held-out 20%):")
    test_scores = {}
    for name, m in fitted_models.items():
        r2, mae, rmse = _metrics_orig(y_test, m.predict(X_test_t))
        _print_metrics(f"{name} test ", r2, mae, rmse)
        test_scores[name] = r2

    # Ensemble evaluation
    print("\n   Ensemble (0.4·XGB + 0.4·LGB + 0.2·GBR):")
    weights = {"XGBoost": 0.40, "LightGBM": 0.40, "GBR": 0.20}
    ens_pred = np.zeros(len(X_test_t))
    total_w  = 0.0
    for name, m in fitted_models.items():
        w         = weights.get(name, 1.0 / len(fitted_models))
        ens_pred += w * m.predict(X_test_t)
        total_w  += w
    ens_pred /= total_w
    ens_r2, ens_mae, ens_rmse = _metrics_orig(y_test, ens_pred)
    _print_metrics("ENSEMBLE test", ens_r2, ens_mae, ens_rmse)

    # Overfit gap report
    print("\n   ── Overfit Gap ────────────────────────────────────────────")
    for name in fitted_models:
        cv_m = cv_results[name][0]
        te   = test_scores[name]
        gap  = cv_m - te
        flag = "✅" if abs(gap) <= 0.03 else ("⚡" if abs(gap) <= 0.06 else "⚠️")
        print(f"      {flag} [{name:<12}]  CV={cv_m:.4f}  "
              f"Test={te:.4f}  Gap={gap:+.4f}")

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    print("\n[7/7] Saving artifacts …")

    ens_members  = [
        (m, weights.get(name, 1.0 / len(fitted_models)), name)
        for name, m in fitted_models.items()
    ]
    ensemble_obj = WeightedEnsemble(preprocessor, ens_members)

    # model.pkl  — WeightedEnsemble (accepts raw pd.DataFrame from app.py)
    joblib.dump(ensemble_obj, MODEL_PATH)

    # scaler.pkl — passthrough (kept for backward-compat, not used in inference)
    joblib.dump(FunctionTransformer(), SCALER_PATH)

    # label_encoders.pkl — metadata dict (no legacy LabelEncoder objects needed)
    joblib.dump({
        "feature_names": feat_names,
        "all_features":  ALL_FEATURES,
        "log_target":    True,
        "best_model":    "WeightedEnsemble",
        "weights":       weights,
        "n_estimators":  best_rounds,
        "version":       "v6",
    }, ENCODERS_PATH)

    print(f"   ✅  model.pkl      → {MODEL_PATH}")
    print(f"   ✅  scaler.pkl     → {SCALER_PATH}")
    print(f"   ✅  encoders.pkl   → {ENCODERS_PATH}")

    # Feature importance
    best_ind = max(test_scores, key=test_scores.get)
    print(f"\n   📊  Feature Importances ({best_ind}):")
    _show_importance(fitted_models[best_ind], col_names)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    if ens_r2 >= 0.90:
        print(f"   🎯  ENSEMBLE Test R² = {ens_r2:.4f}  ✅  ≥ 0.90 ACHIEVED!")
    elif ens_r2 >= 0.85:
        print(f"   ⚡  ENSEMBLE Test R² = {ens_r2:.4f}  (≥ 0.85)")
    else:
        print(f"   ⚠️   ENSEMBLE Test R² = {ens_r2:.4f}")
    print(f"   MAE  = ₹{ens_mae:,.0f}   RMSE = ₹{ens_rmse:,.0f}")
    print(f"   Run  : streamlit run app.py")
    print("=" * 72 + "\n")

    return ensemble_obj, feat_names


if __name__ == "__main__":
    train_best_model()