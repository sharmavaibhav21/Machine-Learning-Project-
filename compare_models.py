"""
compare_models.py  ·  v7  —  4-Model Comparison
================================================
Models
──────
  1. Linear Regression
  2. Random Forest
  3. GradientBoostingRegressor
  4. XGBoost  (early stopping)

Removed: LightGBM, Weighted Ensemble
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, RepeatedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Suppress ALL feature-name / FutureWarnings ─────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("⚠️  pip install xgboost")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    load_and_preprocess, inverse_transform_target,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
)
# Reuse model configs and DataFrame-preserving helpers from train_model
from train_model import (
    _transform_to_df, _build_pipe, _metrics_orig,
    _xgb_config, _gbr_config,
    _train_xgb_early_stop,
    ES_ROUNDS, RANDOM_STATE, TEST_SIZE, VAL_SIZE
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cardekho_dataset.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cv_r2(preprocessor, model_factory, X: pd.DataFrame, y_log,
           n_jobs: int = -1) -> tuple:
    """
    RepeatedKFold 5×2 CV using a fresh Pipeline each fold.
    X is pd.DataFrame throughout — no numpy conversion.
    """
    pipe   = _build_pipe(preprocessor, model_factory())
    rkf    = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y_log, cv=rkf,
                             scoring="r2", n_jobs=n_jobs)
    return scores.mean(), scores.std()


def _eval_row(name: str,
              model,
              X_tr_t: pd.DataFrame, y_tr,
              X_te_t: pd.DataFrame, y_te,
              cv_mean: float, cv_std: float) -> dict:
    tr_r2, _, _      = _metrics_orig(y_tr, model.predict(X_tr_t))
    te_r2, mae, rmse = _metrics_orig(y_te, model.predict(X_te_t))
    gap = cv_mean - te_r2
    return {
        "Model":    name,
        "Train R²": round(tr_r2,   4),
        "CV R²":    round(cv_mean, 4),
        "CV±":      round(cv_std,  4),
        "Test R²":  round(te_r2,   4),
        "Gap":      round(gap,     4),
        "MAE ₹":    int(mae),
        "RMSE ₹":   int(rmse),
    }


def _print_table(df_r: pd.DataFrame):
    W = 92
    print("\n" + "═" * W)
    print("          MODEL COMPARISON  ·  v7  (original ₹ scale — overfit gap tracking)")
    print("═" * W)
    hdr = (f"  {'#':<3}{'Model':<22}{'Train R²':>10}{'CV R²':>9}"
           f"{'CV±':>7}{'Test R²':>10}{'Gap':>8}{'MAE ₹':>12}{'RMSE ₹':>12}")
    print(hdr)
    print("─" * W)
    for _, row in df_r.iterrows():
        r2  = row["Test R²"]
        gap = row["Gap"]
        m_flag = "✅" if r2  >= 0.90 else ("⚡" if r2  >= 0.85 else "⚠️")
        g_flag = "✅" if abs(gap) <= 0.03 else ("⚡" if abs(gap) <= 0.06 else "🔴")
        name   = row["Model"] + " " + m_flag
        print(
            f"  {int(row['Rank']):<3}{name:<24}{row['Train R²']:>10}"
            f"{row['CV R²']:>9}{row['CV±']:>7}{r2:>10}"
            f"{gap:>+8.4f} {g_flag}{row['MAE ₹']:>11,}{row['RMSE ₹']:>12,}"
        )
    print("═" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("   CAR RESALE PRICE  ·  v7  4-MODEL COMPARISON")
    print("   Target: Test R² ≥ 0.90 + Gap ≤ 0.03")
    print("=" * 75)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading & preprocessing …")
    X, y_log, feat_names, preprocessor = load_and_preprocess(DATA_PATH)

    # ── Splits ────────────────────────────────────────────────────────────────
    print("\n[2/4] Splitting data …")
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    print(f"      Pool={len(X_pool):,}  Train={len(X_train):,}  "
          f"Val={len(X_val):,}  Test={len(X_test):,}")

    # ── Fit preprocessor ONCE on X_train (no leakage) ─────────────────────────
    print("\n[3/4] Transforming splits (preprocessor fit on X_train only) …")
    X_train_t = _transform_to_df(preprocessor, X_train, fit=True)
    X_val_t   = _transform_to_df(preprocessor, X_val,   fit=False)
    X_pool_t  = _transform_to_df(preprocessor, X_pool,  fit=False)
    X_test_t  = _transform_to_df(preprocessor, X_test,  fit=False)
    col_names = list(X_train_t.columns)
    print(f"      Transformed feature count : {len(col_names)}")
    print(f"      All splits type           : {type(X_train_t).__name__}  ✅")

    # ── Train all 4 models ────────────────────────────────────────────────────
    print("\n[4/4] Training all models …\n")
    results    = []
    all_models : dict = {}
    best_n    : dict = {}

    # 1. Linear Regression
    print("   [1/4] Linear Regression …")
    lr = LinearRegression()
    lr.fit(X_pool_t, y_pool)
    all_models["Linear Regression"] = lr
    cv_m, cv_s = _cv_r2(preprocessor, LinearRegression, X, y_log)
    results.append(_eval_row("Linear Regression", lr,
                             X_pool_t, y_pool, X_test_t, y_test, cv_m, cv_s))

    # 2. Random Forest (regularised)
    print("   [2/4] Random Forest …")
    rf = RandomForestRegressor(
        n_estimators=600, max_depth=20,
        min_samples_split=10, min_samples_leaf=5,
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_pool_t, y_pool)
    all_models["Random Forest"] = rf
    cv_m, cv_s = _cv_r2(
        preprocessor,
        lambda: RandomForestRegressor(
            n_estimators=600, max_depth=20,
            min_samples_split=10, min_samples_leaf=5,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
        ),
        X, y_log
    )
    results.append(_eval_row("Random Forest", rf,
                             X_pool_t, y_pool, X_test_t, y_test, cv_m, cv_s))

    # 3. GradientBoosting
    print("   [3/4] GradientBoosting …")
    gbr = _gbr_config(n_est=600)
    gbr.fit(X_pool_t, y_pool)
    all_models["GradientBoosting"] = gbr
    cv_m, cv_s = _cv_r2(preprocessor, lambda: _gbr_config(600), X, y_log)
    results.append(_eval_row("GradientBoosting", gbr,
                             X_pool_t, y_pool, X_test_t, y_test, cv_m, cv_s))

    # 4. XGBoost (early stopping)
    if XGB_OK:
        print("   [4/4] XGBoost (early stopping) …")
        _, best_n["xgb"] = _train_xgb_early_stop(
            X_train_t, y_train, X_val_t, y_val
        )
        xgb = _xgb_config(n_est=best_n["xgb"])
        xgb.fit(X_pool_t, y_pool)
        all_models["XGBoost"] = xgb
        cv_m, cv_s = _cv_r2(preprocessor,
                              lambda: _xgb_config(best_n.get("xgb", 800)),
                              X, y_log)
        results.append(_eval_row("XGBoost", xgb,
                                 X_pool_t, y_pool, X_test_t, y_test, cv_m, cv_s))
    else:
        print("   [4/4] XGBoost — SKIPPED (not installed)")

    # ── Print table ───────────────────────────────────────────────────────────
    df_r = pd.DataFrame(results).sort_values("Test R²", ascending=False)
    df_r.insert(0, "Rank", range(1, len(df_r) + 1))
    _print_table(df_r)

    # ── Best model summary ────────────────────────────────────────────────────
    best = df_r.iloc[0]
    r2v  = best["Test R²"]
    print(f"\n🏆  BEST       : {best['Model']}")
    print(f"    Test R²    : {r2v}  "
          f"{'✅ ≥0.90!' if r2v>=0.90 else ('⚡ ≥0.85' if r2v>=0.85 else '⚠️ <0.85')}")
    print(f"    CV R²      : {best['CV R²']} ± {best['CV±']}")
    print(f"    Gap        : {best['Gap']:+.4f}  "
          f"{'✅ tight' if abs(best['Gap'])<=0.03 else '⚡ moderate' if abs(best['Gap'])<=0.06 else '🔴 overfit'}")
    print(f"    MAE        : ₹{best['MAE ₹']:,}")
    print(f"    RMSE       : ₹{best['RMSE ₹']:,}")

    # Feature importances for best tree-based model
    tree_names = ["XGBoost", "GradientBoosting", "Random Forest"]
    best_tree  = next((n for n in df_r["Model"] if n in tree_names), None)
    if best_tree and best_tree in all_models:
        m = all_models[best_tree]
        if hasattr(m, "feature_importances_"):
            imps    = m.feature_importances_
            top_idx = np.argsort(imps)[::-1][:12]
            print(f"\n📊  Feature Importances ({best_tree}):")
            print(f"   {'Feature':<40} {'Score':>8}  Bar")
            print("   " + "─" * 65)
            for i in top_idx:
                n   = col_names[i] if i < len(col_names) else f"feat_{i}"
                bar = "█" * int(imps[i] * 80)
                print(f"   {n:<40} {imps[i]:>8.4f}  {bar}")

    print()
    return df_r


if __name__ == "__main__":
    main()