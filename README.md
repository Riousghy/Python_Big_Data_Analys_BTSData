# BTS On-Time Prediction — Weeks 1–7 (Course Project)

**Student:** Guohao (Rious) Yang  
**Advisor:** Prof. **Yousef Nejatbakhsh**  
**Course:** **Python-Drive Big Data Analysis**  
**Term:** **2025 (Summer)**  
**School:** **Kean University**  
**Repo:** `Python_Big_Data_Analys_BTSData`

Predict U.S. BTS flight delays using a leakage-safe, time-aware pipeline.  
I build baselines (Week 3), add robust features & calibration (Week 4), validate with temporal CV + DL (Week 5), then refine a neural network with proper thresholding & calibration (Week 6 v2), and finally compare everything (Week 7).

---

## Contents
- Project Overview  
- Data, Splits & Leakage Control  
- Tech Stack (detailed)  
- **Week 1–6 Results (Quick Tables)**  
- Weekly Worklog & Results  
  - Week 1 — Data Access & EDA  
  - Week 2 — Feature Engineering v1  
  - Week 3 — Baselines & First Evaluation  
  - Week 4 — Time-Aware Features, Calibration & Ablation  
  - Week 5 — Temporal Cross-Validation & TF MLP  
  - Week 6 v2 — Improved NN + Robust Calibration  
  - Week 7 — Final Comparison & Takeaways  
- How to Reproduce (console-first)  
- Key Artifacts (by week)

> **Note on comparability.** I compare models on the same **TEST window (2025-04)** using **threshold-free** metrics (**PR-AUC**, **ROC-AUC**).  
> CV metrics (Week 5) are reported separately (VAL folds). **Brier** is calibration-sensitive; I always state the calibration used.

---

## Project Overview

- **Tasks**
  - **Classification:** `ArrDel15` (delay > 15 min).  
  - **Regression:** `ArrDelay` (minutes).
- **Split:** Train = **2024**; Val = **2025-Jan..Mar**; Test = **2025-Apr**.
- **Pipeline:** Leakage-safe rolling features & route/carrier profiles computed only from prior history, then merged into Val/Test.
- **Goal:** Achieve a deployable classification system with strong PR-AUC and well-calibrated probabilities (low Brier), plus stable regression of delay minutes.

---

## Data, Splits & Leakage Control

- **Base feature file:** `bts_on_time_data/eda/sample_100k_week2_features.parquet` *(Week-2 output)*.
- **Temporal split (fixed across weeks):**  
  Train = **67,485** (2024) | Val = **24,163** (2025-Jan..Mar) | Test = **8,352** (2025-Apr).
- **Class balance (`ArrDel15`=1):** Train = **0.178**, Val = **0.195**, Test = **0.195**.
- **Leakage control**
  - **Route/Carrier profiles** (median delay, on-time rate, counts) computed on **Train**, then joined.  
  - **Rolling signals** (e.g., `route_on_7d/14d/30d`, `car_on_*`, 30-day medians, hourly 7-day load) computed by **daily rolling with a 1-day lag**, then joined — *no future info*.  
- **Preprocessing:** numeric median imputation + scaling; categorical mode imputation + one-hot; unified with `ColumnTransformer`.

---

## Tech Stack (detailed)

- **Language & OS**
  - Python **3.12** (macOS ARM environment), deterministic seeds via `numpy.random.seed` & `tf.random.set_seed`.
  - Execution is **console-first**; artifacts saved under `bts_on_time_data/eda/`.

- **Data handling**
  - `pandas` (`read_parquet`, merges, groupby, rolling windows), `numpy` (vectorized ops, quantiles).
  - Feature file: Parquet produced in Week-2 (`sample_100k_week2_features.parquet`).

- **Classical ML (scikit-learn)**
  - Models: `LogisticRegression`, `RandomForestClassifier`, `RandomForestRegressor`, `Ridge`.
  - Pipelines: `Pipeline`, `ColumnTransformer` with `SimpleImputer` (median/mode), `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`.
  - Metrics: `roc_auc_score`, `average_precision_score` (PR-AUC), `brier_score_loss`, `precision_recall_curve`, `confusion_matrix`, `r2_score`, `mean_absolute_error`, `mean_squared_error`.
  - Calibration: `IsotonicRegression`; reliability via `calibration_curve`.
  - Cross-validation: temporal folds implemented manually (Week 5 CV artifacts).

- **Deep Learning (TensorFlow / Keras)**
  - TF **2.18** / Keras Sequential MLP: ReLU + He init, **BatchNormalization**, **Dropout=0.3**, **L2** regularization.
  - Training control: `EarlyStopping(monitor="val_prauc", mode="max", restore_best_weights=True)`, `ReduceLROnPlateau`.
  - Class imbalance: `class_weight` from empirical prevalence.
  - Custom **Temperature scaling**: fit scalar **T** on VAL by minimizing BCE; apply via logit rescaling.
  - Thresholding: per-calibration **Val best-F1**; **cost curve** (FP=1, FN=5) to choose operational threshold.

- **Visualization**
  - `matplotlib`: PR curves, reliability diagrams, training curves (ROC/PR over epochs).

- **Project hygiene**
  - All artifacts versioned under `bts_on_time_data/eda/` with per-week prefixes (`week3_*`, `week4_*`, `week5_*`, `week6_v2_*`).
  - Raw data paths (`raw_csv/`) ignored to respect size/licensing.

---

## Week 1–6 Results (Quick Tables)

> **TEST window:** 2025-04, unless explicitly marked as CV (VAL folds).  
> Values are taken from the week artifacts reported in this repo.

### Classification — TEST (threshold-free unless noted)

| Week | Model / Calibration             | PR-AUC | ROC-AUC | Brier | Notes |
|:----:|---------------------------------|:------:|:-------:|:-----:|------|
| 3    | LogReg / DT / RF (baselines)    | ~0.24  | ~0.58   |  N/A  | First pass, uncalibrated, high-recall @best-F1 (prec≈0.225, rec≈0.75). |
| 4    | Best classical (RF/HGB)         | ~0.247 | ~0.589  | ~0.156–0.157 | Reliability & calibration plots produced. |
| 5    | **TF MLP (TEST)**               | **0.9204** | **0.9609** | **0.0446** | `week5_tf_prob_test.csv`; cm@0.5=TP1259/FP99/TN6537/FN373; @best Val-F1 thr≈0.448. |
| 6 v2 | **NN (uncalibrated)**           | **0.9228** | **0.9626** | 0.0550 | cm@0.5=TP1426/FP403/TN6317/FN206; @best Val-F1 thr≈0.816. |
| 6 v2 | **NN + Isotonic**               | 0.9183 | 0.9626 | **0.0427** | Best Brier; best-F1 thr on Val≈0.479. |
| 6 v2 | NN + Temperature (T≈0.929)      | 0.9228 | 0.9626 | 0.0552 | Matches uncal PR/ROC; does not improve Brier here. |

### Regression — TEST

| Week | Model     | RMSE   | MAE   | R²     | Notes |
|:----:|-----------|:------:|:-----:|:------:|------|
| 3    | RF        | 11.795 | 8.341 | 0.9494 | First stable baseline. |
| 4    | **Ridge** | **11.056** | **7.744** | **0.9560** | Best linear baseline; improved with time-aware features. |

### Week 5 — Temporal CV (VAL folds)

| Task | Model | PR-AUC | ROC-AUC | Brier | Notes |
|:----:|:-----:|:------:|:-------:|:-----:|------|
| Clf  | LogReg | **0.9223** | **0.9621** | 0.0651 | Mean over temporal folds (`week5_cv_classification.csv`). |
| Clf  | RF     | 0.9050 | 0.9559 | **0.0536** | Strong, slightly worse PR-AUC vs LogReg. |
| Reg  | Ridge  | — | — | — | RMSE **11.236**, MAE **8.022**, R² **0.9600** (fold mean). |
| Reg  | RF     | — | — | — | RMSE **11.740**, MAE **8.366**, R² **0.9563** (fold mean). |

---

## Weekly Worklog & Results

### Week 1 — Data Access & EDA
- **Goal:** Obtain BTS on-time data, basic cleaning, sanity checks.  
- **Outputs:** Column auditing, missingness, initial class balance used to guide later feature design.

---

### Week 2 — Feature Engineering v1
- **Goal:** Build time/calendar/profile features; save a reproducible feature file.  
- **Output:** `bts_on_time_data/eda/sample_100k_week2_features.parquet` *(used by all later weeks)*.
- **Key features (examples)**
  - **Core time:** `Distance`, `CRSDepTime_min`, `dow`, `is_weekend`, `quarter`, `season`, `tod_bin`  
  - **Profiles:** `route_med_arr_delay`, `route_ontime_rate`, `route_flights`, `carrier_*`  
  - **Rolling:** `route_on_7d/14d/30d`, `route_med_30d`, `car_on_*`, `car_med_30d`  
  - **Hourly load (7D):** `orig_hour_load_7d`, `dest_hour_load_7d`  
  - **Optional recency:** `DepDelay`, `TaxiOut` (if present)

---

### Week 3 — Baselines & First Evaluation

**Classification (TEST, uncalibrated)**  
- Overall separability **moderate**: **ROC-AUC ≈ 0.58**, **PR-AUC ≈ 0.24**.  
- Best-F1 (TEST) shows **recall ≈ 0.75**, **precision ≈ 0.225** across LogReg/DT/RF → a high-recall alerting baseline.  
- **Artifacts:** `week3_clf_prob_logreg.csv`, `week3_clf_prob_dt.csv`, `week3_clf_prob_rf.csv` (+ PR PNGs), `week3_metrics.json`.

**Regression (TEST)**  
- **Random Forest:** **MAE 8.341**, **RMSE 11.795**, **R² 0.9494**.  
- **Linear Regression** unstable → pivot to **Ridge** later.  
- **Artifacts:** `week3_reg_pred_linreg.csv`, `week3_reg_pred_rf.csv`.

---

### Week 4 — Time-Aware Features, Calibration & Ablation

**Regression (TEST)**  
- **Ridge:** **RMSE 11.056**, **MAE 7.744**, **R² 0.956** *(best so far)*  
- **RF:** **RMSE 11.741**, **MAE 8.287**, **R² 0.950**  
- **Artifacts:** `week4_reg_pred_ridge.csv`, `week4_reg_pred_rf.csv`.

**Classification (TEST)**  
- **LogReg:** **ROC-AUC 0.582**, **PR-AUC 0.241**, **Brier 0.157**  
- **RF / HGB:** **ROC-AUC ≈ 0.589**, **PR-AUC ≈ 0.247**, **Brier ≈ 0.156–0.157**  
- **Ablation:** `week4_ablation_{clf,reg}.csv`  
- **Curves:** `week4_reliability_*png`, `week4_clf_pr_*png`  
- **Backtests:** `week4_backtest_{clf,reg}.csv`  
- **Summary:** `week4_metrics.json`

**Takeaway:** Time-aware features + calibration landed; regression improved; classification awaited stronger non-linear capacity.

---

### Week 5 — Temporal Cross-Validation & TF MLP

**Temporal CV (VAL folds)** — `week5_cv_{classification,regression}.csv`  
- **LogReg (mean):** **ROC-AUC 0.9621**, **PR-AUC 0.9223**, **Brier 0.0651**  
- **RF (mean):** **ROC-AUC 0.9559**, **PR-AUC 0.9050**, **Brier 0.0536**  
- **Ridge (Reg, mean):** **RMSE 11.236**, **MAE 8.022**, **R² 0.9600**

**TF MLP (TEST)** — `week5_tf_prob_test.csv`  
- **ROC-AUC 0.9609**, **PR-AUC 0.9204**, **Brier 0.0446**  
- **Confusion (TEST)**  
  - **@0.5:** **TP 1259**, **FP 99**, **TN 6537**, **FN 373**  
  - **@best (Val F1 thr ≈ 0.448):** **TP 1287**, **FP 115**, **TN 6521**, **FN 345**  
- **Curves:** `week5_tf_training_{roc,pr}.png`, `week5_tf_{pr,reliability}_test.png`  
- **Bundle:** `week5_metrics.json`

**Takeaway:** With leakage-safe features and temporal CV, linear LogReg is already very strong on VAL; a small TF MLP matches it on TEST and improves Brier.

---

### Week 6 v2 — Improved NN + Robust Calibration

**Changes**  
- Deeper **MLP** (128-64-32, ReLU, **BN**, **Dropout=0.3**, **L2**), `class_weight` from prevalence, **EarlyStopping** on Val **PR-AUC**, **ReduceLROnPlateau**.  
- **Consistent thresholding** per-calibration (Val best-F1) and **cost curve** (FP=1, FN=5) for operations.  
- **Calibration:** **Isotonic** (Val-fit) & **Temperature scaling** (Val-fit).

**TEST metrics (2025-04)** — `week6_v2_metrics.json`, `week6_v2_prob_test.csv`  
- **NN (uncalibrated):** **PR-AUC 0.9228**, **ROC-AUC 0.9626**, **Brier 0.055**  
  - **cm@0.5:** **TP 1426**, **FP 403**, **TN 6317**, **FN 206**  
  - **cm@best (Val-F1 thr ≈ 0.816):** **TP 1305**, **FP 124**, **TN 6596**, **FN 327**  
- **NN + Isotonic:** **PR-AUC 0.9183**, **ROC-AUC 0.9626**, **Brier 0.0427** *(best Brier)*  
- **NN + T-scale (T≈0.929):** **PR-AUC 0.9228**, **ROC-AUC 0.9626**, **Brier 0.0552**

**Cost-aware threshold (Val)** — `week6_v2_cost_curve_val.csv`  
- **Best thr ≈ 0.465**, **TEST cost ≈ 1440**, **cm@thr:** **TP 1433**, **FP 445**, **TN 6275**, **FN 199**

**Backtest** — `week6_v2_backtest_dl.csv`  
- Month-by-month **PR-AUC / ROC-AUC / Brier** across late-2024 & 2025-01..04 shows stability.

**Curves**  
- PR: `week6_v2_pr_{uncal,iso,T}.png`  
- Reliability: `week6_v2_reliability_{uncal,iso,T}.png`  
- Training: `week6_v2_training_{roc,pr}.png`

**Takeaway:** Week 6 v2 slightly **edges Week 5** on TEST **PR-AUC** (**0.9228 vs 0.9204**) while **Isotonic** yields **best Brier (~0.0427)** → better probability quality for decisioning.

---

### Week 7 — Final Comparison & Takeaways

**Comparable TEST (threshold-free) — Classification**  
- **Week 6 v2 (NN, uncal):** **PR-AUC 0.9228**, **ROC-AUC 0.9626**  
- **Week 6 v2 (NN, isotonic):** **PR-AUC 0.9183**, **ROC-AUC 0.9626**, **Brier 0.0427 (best)**  
- **Week 5 (TF MLP):** **PR-AUC 0.9204**, **ROC-AUC 0.9609**, **Brier 0.0446**  
- **Week 4 (LogReg/RF/HGB, TEST):** **PR-AUC ≈ 0.241–0.247**, **ROC-AUC ≈ 0.582–0.589**, **Brier ≈ 0.156–0.157**  
- **Week 3 (Baselines, TEST):** **PR-AUC ≈ 0.24**, **ROC-AUC ≈ 0.58**

**Regression (TEST)**  
- **Week 4 — Ridge:** **RMSE 11.056**, **MAE 7.744**, **R² 0.956** *(best)*  
- **Week 3 — RF:** **RMSE 11.795**, **MAE 8.341**, **R² 0.9494**

**CV (VAL folds — Week 5 robustness)**  
- **LogReg:** **PR-AUC 0.9223**, **ROC-AUC 0.9621**, **Brier 0.0651**  
- **RF:** **PR-AUC 0.9050**, **ROC-AUC 0.9559**, **Brier 0.0536**

**What improved and why**  
1) **Leakage-safe features** (Weeks 2–4): rolling route/carrier/traffic signals with 1-day lag → better structure.  
2) **Temporal validation** (Week 5): realistic generalization; shows LogReg’s ceiling with current features.  
3) **Neural network training** (Week 6 v2): right capacity + class weighting + early stopping + LR schedule → slight PR-AUC gain on TEST.  
4) **Calibration:** Isotonic cuts **Brier** from ~0.055 → **~0.043**, improving decision quality.  
5) **Thresholding:** per-calibration Val **best-F1** and **cost-aware** thresholds provide clear operating points.

---
