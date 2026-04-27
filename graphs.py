"""
graphs.py
=========
Generates and saves all project visualizations as high-resolution
PNG files. Run this script independently after data is available.

Generated files:
  01_price_distribution.png
  02_car_age_vs_price.png
  03_km_driven_vs_price.png
  04_correlation_heatmap.png
  05_model_comparison.png
  06_feature_importance.png
  07_fuel_type_vs_price.png
  08_transmission_vs_price.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_preprocess

DATA_PATH  = os.path.join(os.path.dirname(__file__), "cardekho_dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#1A1A2E",
    "accent1":   "#E94560",
    "accent2":   "#0F3460",
    "accent3":   "#533483",
    "accent4":   "#16213E",
    "highlight": "#FFD700",
    "bg":        "#F8F9FA",
}
MODEL_COLORS = ["#E94560", "#0F3460", "#533483", "#16A085"]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   COLORS["bg"],
    "axes.facecolor":     COLORS["bg"],
    "axes.labelcolor":    COLORS["primary"],
    "xtick.color":        COLORS["primary"],
    "ytick.color":        COLORS["primary"],
    "text.color":         COLORS["primary"],
})


def save(name: str, fig=None):
    path = os.path.join(OUTPUT_DIR, name)
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight",
                         facecolor=COLORS["bg"])
    plt.close("all")
    print(f"   \u2705  Saved \u2192 {path}")


def load_raw():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    for col in ("car_name", "model", "name"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


# ── 1. Price Distribution ──────────────────────────────────────────────────────
def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Car Selling Price Distribution", fontsize=16,
                 fontweight="bold", color=COLORS["primary"])

    axes[0].hist(df["selling_price"] / 1e5, bins=60, color=COLORS["accent1"],
                 edgecolor="white", linewidth=0.4, alpha=0.9)
    axes[0].set_xlabel("Selling Price (Lakhs \u20b9)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Raw Distribution")

    log_prices = np.log1p(df["selling_price"])
    axes[1].hist(log_prices, bins=60, color=COLORS["accent2"],
                 edgecolor="white", linewidth=0.4, alpha=0.9)
    axes[1].set_xlabel("log(Selling Price + 1)")
    axes[1].set_title("Log-Transformed Distribution")

    plt.tight_layout()
    save("01_price_distribution.png", fig)


# ── 2. Car Age vs Price ────────────────────────────────────────────────────────
def plot_car_age_vs_price(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        df["vehicle_age"], df["selling_price"] / 1e5,
        c=df["selling_price"], cmap="RdYlGn_r",
        alpha=0.5, s=20, linewidths=0
    )
    plt.colorbar(scatter, ax=ax, label="Selling Price (\u20b9)")
    ax.set_xlabel("Car Age (Years)")
    ax.set_ylabel("Selling Price (Lakhs \u20b9)")
    ax.set_title("Car Age vs Selling Price", fontsize=14, fontweight="bold")

    z = np.polyfit(df["vehicle_age"], df["selling_price"] / 1e5, 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["vehicle_age"].min(), df["vehicle_age"].max(), 200)
    ax.plot(x_line, p(x_line), color=COLORS["accent1"],
            linewidth=2.5, linestyle="--", label="Trend")
    ax.legend()
    plt.tight_layout()
    save("02_car_age_vs_price.png", fig)


# ── 3. KM Driven vs Price ──────────────────────────────────────────────────────
def plot_km_vs_price(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df["km_driven"] / 1000, df["selling_price"] / 1e5,
               color=COLORS["accent3"], alpha=0.4, s=18, linewidths=0)
    ax.set_xlabel("KM Driven (Thousands)")
    ax.set_ylabel("Selling Price (Lakhs \u20b9)")
    ax.set_title("KM Driven vs Selling Price", fontsize=14, fontweight="bold")
    ax.set_xlim(0, df["km_driven"].quantile(0.99) / 1000)
    ax.set_ylim(0, df["selling_price"].quantile(0.99) / 1e5)
    plt.tight_layout()
    save("03_km_driven_vs_price.png", fig)


# ── 4. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap():
    X, y, feature_names, _ = load_and_preprocess(DATA_PATH)
    df_corr = X.select_dtypes(include=["number"]).copy()
    df_corr["selling_price"] = y.values
    corr = df_corr.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save("04_correlation_heatmap.png", fig)


# ── 5. Model Comparison ───────────────────────────────────────────────────────
def plot_model_comparison():
    X, y, _, _ = load_and_preprocess(DATA_PATH)
    X = X.select_dtypes(include=["number"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        "Linear\nRegression": (LinearRegression(), X_tr_s, X_te_s),
        "Decision\nTree":     (DecisionTreeRegressor(max_depth=12, random_state=42),
                               X_train, X_test),
        "Random\nForest":     (RandomForestRegressor(n_estimators=300, max_depth=25,
                                                     random_state=42, n_jobs=-1),
                               X_train, X_test),
        "SVR":                (SVR(kernel="rbf", C=100, gamma="scale"), X_tr_s, X_te_s),
    }

    names, r2_scores = [], []
    for name, (m, Xtr, Xte) in models.items():
        m.fit(Xtr, y_train)
        r2_scores.append(r2_score(y_test, m.predict(Xte)))
        names.append(name)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, r2_scores, color=MODEL_COLORS, width=0.5,
                  edgecolor="white", linewidth=1.2)
    ax.axhline(0.90, color=COLORS["accent1"], linestyle="--",
               linewidth=2, label="Target R\u00b2 = 0.90")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R\u00b2 Score")
    ax.set_title("Model Comparison \u2014 R\u00b2 Score", fontsize=14, fontweight="bold")
    ax.legend()
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold",
                fontsize=11)
    plt.tight_layout()
    save("05_model_comparison.png", fig)


# ── 6. Feature Importance ─────────────────────────────────────────────────────
def plot_feature_importance():
    X, y, feature_names, _ = load_and_preprocess(DATA_PATH)
    X = X.select_dtypes(include=["number"])
    rf = RandomForestRegressor(n_estimators=300, max_depth=25,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imps = rf.feature_importances_
    sorted_idx = np.argsort(imps)
    feats = [X.columns[i] for i in sorted_idx]
    vals  = imps[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feats, vals, color=COLORS["accent2"], edgecolor="white")
    bars[-1].set_color(COLORS["accent1"])
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest \u2014 Feature Importances",
                 fontsize=14, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    save("06_feature_importance.png", fig)


# ── 7. Fuel Type vs Price ─────────────────────────────────────────────────────
def plot_fuel_vs_price(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fuel_groups = df.groupby("fuel_type")["selling_price"].median() / 1e5
    fuel_groups.sort_values(ascending=False).plot(
        kind="bar", ax=ax, color=MODEL_COLORS[:len(fuel_groups)],
        edgecolor="white", rot=0
    )
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Median Selling Price (Lakhs \u20b9)")
    ax.set_title("Fuel Type vs Median Selling Price", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save("07_fuel_type_vs_price.png", fig)


# ── 8. Transmission vs Price ──────────────────────────────────────────────────
def plot_transmission_vs_price(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    trans_groups = df.groupby("transmission_type")["selling_price"]
    data_to_plot = [group.values / 1e5 for _, group in trans_groups]
    labels       = list(trans_groups.groups.keys())

    ax.boxplot(data_to_plot, tick_labels=labels, notch=False, patch_artist=True,
               boxprops=dict(facecolor=COLORS["accent2"], color=COLORS["primary"]),
               medianprops=dict(color=COLORS["accent1"], linewidth=2))
    ax.set_ylabel("Selling Price (Lakhs \u20b9)")
    ax.set_xlabel("Transmission Type")
    ax.set_title("Transmission Type vs Selling Price", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save("08_transmission_vs_price.png", fig)


def main():
    print("=" * 55)
    print("   GENERATING ALL VISUALIZATIONS")
    print("=" * 55)

    print("\nLoading raw dataset \u2026")
    df = load_raw()
    print(f"   Rows: {len(df)}")

    print("\nGenerating graphs:")
    plot_price_distribution(df)
    plot_car_age_vs_price(df)
    plot_km_vs_price(df)
    plot_correlation_heatmap()
    plot_model_comparison()
    plot_feature_importance()
    plot_fuel_vs_price(df)
    plot_transmission_vs_price(df)

    print(f"\n\U0001f389  All 8 graphs saved to: {OUTPUT_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
