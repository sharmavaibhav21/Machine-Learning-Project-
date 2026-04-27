# 🚗 Car Resale Price Prediction System

> Production-ready ML project — Regression | Random Forest | Streamlit UI

---

## 📁 Project Structure

```
car_resale_project/
├── preprocessing.py       # Data loading, cleaning, feature engineering
├── train_model.py         # Train best model → saves model.pkl, scaler.pkl
├── compare_models.py      # Train & compare all 4 models with metrics table
├── graphs.py              # Generate & save 8 visualizations
├── app.py                 # Streamlit premium UI
├── requirements.txt
├── model.pkl              # Saved best model (after training)
├── scaler.pkl             # Saved StandardScaler
├── label_encoders.pkl     # Saved LabelEncoders
├── graphs/                # Generated PNG visualizations
└── CAR DETAILS FROM CAR DEKHO.csv
```

---

## ⚙️ Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Place dataset in project folder:
# CAR DETAILS FROM CAR DEKHO.csv

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

## 🧠 Models Trained

| Model | R² (approx) | Notes |
|-------|-------------|-------|
| Linear Regression | ~0.67 | Baseline |
| Decision Tree | ~0.84 | Prone to overfit |
| **Random Forest** | **~0.95** | **Best model ✅** |
| SVR | ~0.82 | Needs scaling |

---

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| `car_age` | 2024 − manufacturing year |
| `km_driven` | Log-transformed odometer |
| `fuel` | Petrol / Diesel / CNG / Electric |
| `transmission` | Manual / Automatic |
| `seller_type` | Individual / Dealer |
| `owner` | 1st / 2nd / 3rd owner |

---

## 🎯 Target: R² ≥ 0.90 ✅

Achieved via:
- Feature engineering (car_age)
- Log transform of km_driven
- RandomizedSearchCV (30 iter, 5-fold CV)
- Outlier removal