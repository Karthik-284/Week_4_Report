# Predictive Modeling and Optimization in Logistics Systems

**Week 4 Task — Logistics Data Analyst Internship**

Forecasts shipment **delivery time (hours)** from operational features (distance, stops, traffic, weather, carrier type, etc.) using machine learning, then turns the model's feature importances into concrete logistics optimization recommendations.

Full write-up with results, charts, and discussion: see `Week4_Predictive_Modeling_Logistics_Report.docx`.

## Problem

Predict `delivery_time_hours` for a shipment as a supervised regression problem, then use the trained model to identify which operational levers (route distance, carrier choice, traffic, warehouse processing time, etc.) have the greatest effect on delivery performance — and recommend how to act on them.

## Dataset

`shipments.csv` — 2,400 simulated shipment records (no proprietary company data was available for this exercise, so a realistic synthetic dataset was generated in NumPy with relationships mirroring real logistics operations — see Section 2.2 of the report for details).

| Feature | Type | Description |
|---|---|---|
| distance_km | numeric | Route distance, origin to destination |
| package_weight_kg | numeric | Weight of the shipped package |
| num_stops | numeric | Number of intermediate stops/hubs |
| traffic_index | numeric | Congestion severity, 0.5 (free flow) – 3.0 (heavy) |
| warehouse_processing_hrs | numeric | Time in warehouse before dispatch |
| fuel_price_index | numeric | Regional fuel price index |
| is_peak_season | binary | 1 if shipped during peak demand |
| carrier_type | categorical | Road / Rail / Air |
| weather_condition | categorical | Clear / Rain / Storm |
| **delivery_time_hours** | numeric | **Target** — total delivery time |

## Approach

1. **Preprocessing** — numeric features standardized, categorical features one-hot encoded, wrapped in a scikit-learn `ColumnTransformer` + `Pipeline` to prevent data leakage.
2. **Models compared** — Linear Regression (interpretable baseline), Decision Tree, Random Forest (ensemble).
3. **Validation** — 80/20 train-test split plus 5-fold cross-validation.
4. **Tuning** — `GridSearchCV` over Random Forest hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`).
5. **Optimization** — feature importances from the tuned Random Forest translated into prioritized, actionable recommendations (routing, carrier mix, congestion-aware dispatch, warehouse throughput, peak-season staffing).

## Results

| Model | RMSE (hrs) | MAE (hrs) | R² | 5-Fold CV R² |
|---|---|---|---|---|
| Linear Regression | 1.83 | 1.43 | 0.851 | 0.886 |
| Decision Tree | 2.53 | 2.01 | 0.713 | 0.763 |
| Random Forest (default) | 1.82 | 1.47 | 0.851 | 0.879 |
| Random Forest (tuned) | 1.82 | 1.47 | 0.851 | 0.879 |

Top predictive drivers of delivery time: **route distance (~66%)**, **carrier type (~14% combined)**, and **traffic congestion (~8%)**.

## Repo contents

```
├── pipeline.py    # full pipeline: data simulation, preprocessing, training,
│                  # cross-validation, hyperparameter tuning, chart generation
├── shipments.csv  # simulated dataset (2,400 rows)
└── README.md
```

## Run it

```bash
pip install numpy pandas matplotlib scikit-learn
python pipeline.py
```

Outputs: trained model metrics printed to console, plus `feature_importance.png`, `actual_vs_predicted.png`, `model_comparison.png`, `model_summary.csv`, and `feature_importance.csv`.
