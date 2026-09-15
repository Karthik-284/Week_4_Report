"""
Week 4 Task: Predictive Modeling and Optimization in Logistics Systems
Problem: Forecast shipment delivery time (in hours) using operational features,
then translate model insights into optimization recommendations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RNG = np.random.default_rng(42)
N = 2400

# ---------------------------------------------------------------------------
# 1. DATA SIMULATION
# ---------------------------------------------------------------------------
distance_km = RNG.gamma(shape=3.0, scale=60, size=N).clip(5, 900)
package_weight_kg = RNG.gamma(shape=2.0, scale=8, size=N).clip(0.5, 200)
num_stops = RNG.poisson(3, size=N).clip(0, 12)
traffic_index = RNG.uniform(0.5, 3.0, size=N)          # 1 = free flow, 3 = heavy congestion
warehouse_processing_hrs = RNG.gamma(shape=2.0, scale=0.8, size=N).clip(0.1, 8)
fuel_price_index = RNG.normal(100, 12, size=N).clip(70, 140)
is_peak_season = RNG.choice([0, 1], size=N, p=[0.75, 0.25])
carrier_type = RNG.choice(["Road", "Rail", "Air"], size=N, p=[0.65, 0.20, 0.15])
weather_condition = RNG.choice(["Clear", "Rain", "Storm"], size=N, p=[0.7, 0.22, 0.08])

carrier_speed_factor = pd.Series(carrier_type).map({"Road": 1.0, "Rail": 0.75, "Air": 0.35}).values
weather_delay_factor = pd.Series(weather_condition).map({"Clear": 0.0, "Rain": 1.2, "Storm": 3.5}).values

base_hours = 1.5 + distance_km * 0.045 * carrier_speed_factor
stop_delay = num_stops * 0.35
traffic_delay = traffic_index * 1.8
weight_delay = np.log1p(package_weight_kg) * 0.25
season_delay = is_peak_season * 2.1
noise = RNG.normal(0, 1.4, size=N)

delivery_time_hours = (
    base_hours + stop_delay + traffic_delay + weight_delay
    + weather_delay_factor + season_delay + warehouse_processing_hrs * 0.6 + noise
).clip(1, None)

df = pd.DataFrame({
    "distance_km": distance_km,
    "package_weight_kg": package_weight_kg,
    "num_stops": num_stops,
    "traffic_index": traffic_index,
    "warehouse_processing_hrs": warehouse_processing_hrs,
    "fuel_price_index": fuel_price_index,
    "is_peak_season": is_peak_season,
    "carrier_type": carrier_type,
    "weather_condition": weather_condition,
    "delivery_time_hours": delivery_time_hours,
})

df.to_csv("shipments.csv", index=False)
print("Dataset shape:", df.shape)
print(df.describe(include="all").T)

# ---------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT & PREPROCESSING
# ---------------------------------------------------------------------------
target = "delivery_time_hours"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ["distance_km", "package_weight_kg", "num_stops", "traffic_index",
                     "warehouse_processing_hrs", "fuel_price_index", "is_peak_season"]
categorical_features = ["carrier_type", "weather_condition"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features),
])

# ---------------------------------------------------------------------------
# 3. MODEL DEFINITIONS
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
}

results = {}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring="r2")

    results[name] = {
        "pipe": pipe, "rmse": rmse, "mae": mae, "r2": r2,
        "cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std(),
    }
    print(f"\n{name}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  "
          f"CV R2={cv_scores.mean():.3f}+/-{cv_scores.std():.3f}")

# ---------------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (Random Forest — best base performer)
# ---------------------------------------------------------------------------
rf_pipe = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(random_state=42))])
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 8, 12, 16],
    "model__min_samples_leaf": [1, 2, 4],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
best_preds = best_rf.predict(X_test)
best_rmse = mean_squared_error(y_test, best_preds) ** 0.5
best_mae = mean_absolute_error(y_test, best_preds)
best_r2 = r2_score(y_test, best_preds)

print("\nBest RF params:", grid.best_params_)
print(f"Tuned RF -> RMSE={best_rmse:.3f}  MAE={best_mae:.3f}  R2={best_r2:.3f}")
print(f"Best CV RMSE during grid search: {-grid.best_score_:.3f}")

tuned_cv_r2 = cross_val_score(best_rf, X_train, y_train, cv=kfold, scoring="r2")

results["Random Forest (Tuned)"] = {
    "pipe": best_rf, "rmse": best_rmse, "mae": best_mae, "r2": best_r2,
    "cv_r2_mean": tuned_cv_r2.mean(), "cv_r2_std": tuned_cv_r2.std(),
}

# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (from tuned Random Forest)
# ---------------------------------------------------------------------------
ohe = best_rf.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_names
importances = best_rf.named_steps["model"].feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nFeature importances:\n", imp_series)

# ---------------------------------------------------------------------------
# 6. CHARTS
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 4.2))
imp_series.plot(kind="barh", color="#D97757")
plt.gca().invert_yaxis()
plt.title("Feature Importance — Tuned Random Forest")
plt.xlabel("Relative importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

plt.figure(figsize=(5.2, 5.2))
plt.scatter(y_test, best_preds, alpha=0.4, color="#D97757", edgecolor="none")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel("Actual delivery time (hrs)")
plt.ylabel("Predicted delivery time (hrs)")
plt.title("Actual vs. Predicted — Tuned Random Forest")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

model_names = list(results.keys())
rmse_vals = [results[m]["rmse"] for m in model_names]
plt.figure(figsize=(7, 4.2))
plt.bar(model_names, rmse_vals, color=["#8C8C8C", "#8C8C8C", "#8C8C8C", "#D97757"])
plt.ylabel("Test RMSE (hours)")
plt.title("Model Comparison — Test RMSE (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. SAVE SUMMARY TABLE FOR REPORT
# ---------------------------------------------------------------------------
summary_rows = []
for name, r in results.items():
    summary_rows.append({
        "Model": name, "RMSE": round(r["rmse"], 3), "MAE": round(r["mae"], 3),
        "R2": round(r["r2"], 3), "CV_R2_mean": round(r["cv_r2_mean"], 3),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("model_summary.csv", index=False)
print("\n", summary_df)

imp_series.round(4).to_csv("feature_importance.csv")

print("\nBest hyperparameters:", grid.best_params_)
print("DONE")
