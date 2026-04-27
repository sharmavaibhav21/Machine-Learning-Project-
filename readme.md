# 🚗 Car Resale Price Prediction System

> Production-ready ML project — GradientBoosting | XGBoost | Streamlit UI

---

## 📁 Project Structure

```
car_resale_project/
├── preprocessing.py       # Data loading, cleaning, feature engineering
├── train_model.py         # Train best model → saves model.pkl, preprocessor.pkl
├── compare_models.py      # Train & compare all 4 models with metrics table
├── graphs.py              # Generate & save 8 visualizations
├── app.py                 # Streamlit premium UI
├── requirements.txt
├── model.pkl              # Saved best model (GradientBoosting)
├── preprocessor.pkl       # Saved fitted ColumnTransformer (used by app.py)
├── scaler.pkl             # Passthrough scaler (backward-compat)
├── label_encoders.pkl     # Metadata dict (feature names, best model, version)
├── graphs/                # Generated PNG visualizations
└── cardekho_dataset.csv
```

---

## ⚙️ Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Place dataset in project folder:
# cardekho_dataset.csv

# Step 1 — Train the best model
python train_model.py

# Step 2 — Compare all 4 models
python compare_models.py

# Step 3 — Generate visualizations
python graphs.py

# Step 4 — Launch Streamlit app
streamlit run app.py
```

---

## 🧠 Models Trained & Compared

| # | Model | Train R² | CV R² | Test R² | Gap | MAE ₹ | RMSE ₹ |
|---|-------|----------|-------|---------|-----|-------|--------|
| 1 | **GradientBoosting ✅** | **0.9615** | **0.9383** | **0.9426** | **−0.0043** | **₹92,200** | **₹1,75,687** |
| 2 | XGBoost ✅ | 0.9551 | 0.9374 | 0.9386 | −0.0013 | ₹94,822 | ₹1,81,722 |
| 3 | Random Forest ⚡ | 0.9275 | 0.9297 | 0.8984 | +0.0313 | ₹1,05,309 | ₹2,33,776 |
| 4 | Linear Regression ⚠️ | 0.6860 | 0.9044 | 0.8582 | +0.0461 | ₹1,28,106 | ₹2,76,175 |

> ✅ = Test R² ≥ 0.90 · ⚡ = Test R² ≥ 0.85 · Gap ✅ = |CV − Test| ≤ 0.03

**Best Model: GradientBoosting** — Test R² = **0.9426**, MAE = ₹92,200, zero overfit.

---

## 📊 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `vehicle_age` | Numerical | 2024 − manufacturing year |
| `km_driven_log` | Numerical | log1p(km_driven) — compresses right skew |
| `mileage` | Numerical | Fuel efficiency (km/l) |
| `engine` | Numerical | Displacement in CC |
| `max_power` | Numerical | Peak power in BHP |
| `km_per_year` | Numerical | km_driven_log / (vehicle_age + 1) |
| `power_to_engine` | Numerical | max_power / engine — performance density |
| `age_power_interaction` | Numerical | vehicle_age × max_power — depreciation signal |
| `brand` | Categorical | Top 22 brands; rare → "Other" |
| `fuel_type` | Categorical | Petrol / Diesel / CNG / LPG / Electric |
| `seller_type` | Categorical | Individual / Dealer / Trustmark Dealer |
| `transmission_type` | Categorical | Manual / Automatic |
| `seats_cat` | Categorical | 5_or_less / 6_7 / 8_plus |

---

## 🏆 Top Feature Importances (GradientBoosting)

| Rank | Feature | Score |
|------|---------|-------|
| 1 | `max_power` | 0.4224 |
| 2 | `engine` | 0.1636 |
| 3 | `km_per_year` | 0.1259 |
| 4 | `vehicle_age` | 0.1173 |
| 5 | `transmission_type_Automatic` | 0.0394 |
| 6 | `power_to_engine` | 0.0360 |
| 7 | `km_driven_log` | 0.0196 |

---

## 🎯 Results Summary

```
Dataset      : 15,411 rows → 15,409 after outlier clip (99.5th pct)
Train pool   : 12,327  |  Train: 10,477  |  Val: 1,850  |  Test: 3,082
Features     : 13 input → 44 transformed (after OHE + scaling)

Best Model   : GradientBoosting  (n_estimators=600, max_depth=5, lr=0.03)
Test R²      : 0.9426  ✅  ≥ 0.90 ACHIEVED
CV R²        : 0.9383 ± 0.0021  (5×2 RepeatedKFold)
Overfit Gap  : −0.0043  ✅  well within 0.03 limit
MAE          : ₹92,200
RMSE         : ₹1,75,687

XGBoost      : Test R² = 0.9386  (early stop @ round 969)
```

---

## ⚙️ Pipeline Architecture

```
Raw CSV
  └─ dropna / drop junk columns
  └─ clip outliers (99.5th pct — no rows dropped)
  └─ group rare brands → top 22 + "Other"
  └─ feature engineering (8 numerical features)
       ↓
  ColumnTransformer (fit on X_train only — zero leakage)
    ├─ StandardScaler  →  8 numerical features
    └─ OneHotEncoder   →  5 categorical features  →  36 dummy columns
       ↓  (44 total features)
  Model training
    ├─ XGBoost  : early stopping on val set → best n_estimators = 969
    └─ GBR      : fixed n_estimators = 600
       ↓
  Best model (GBR) + preprocessor saved as model.pkl + preprocessor.pkl
       ↓
  app.py: raw pd.DataFrame → preprocessor.pkl → model.predict()
```

---

## 🗃️ Saved Artifacts

| File | Contents |
|------|----------|
| `model.pkl` | Best fitted model (GradientBoosting) |
| `preprocessor.pkl` | Fitted ColumnTransformer — **required by app.py** |
| `scaler.pkl` | Passthrough (backward-compat) |
| `label_encoders.pkl` | Metadata dict: feature names, best model name, version |