"""5-Fold Cross-Validation Model Ablation Benchmark Suite.

Evaluates 5 Model Families:
1. Logistic Regression (Linear Baseline)
2. MLPClassifier (Neural Network Baseline)
3. Random Forest (Bagging Baseline)
4. BaselineXGBPredictor (GBDT v1 Prototype)
5. CalibratedTriadPredictor (XGBoost + LightGBM + CatBoost with Isotonic Calibration)

Evaluates:
- Macro-F1 Score
- Multi-Class ROC-AUC (One-vs-Rest Macro)
- Multi-Class Brier Calibration Score
- Expected Calibration Error (ECE - 10 Bins)
- p99 CPU Inference Latency (ms)

Exports results to artifacts/benchmark_results.json and persists trained production models.
"""

import json
import os
import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

from app.services.risk_engine.calibrated_triad import CalibratedTriadPredictor
from app.services.risk_engine.constants import (
    SUPPORTED_DISEASES,
)
from app.services.risk_engine.xgb_baseline import BaselineXGBPredictor


def generate_agronomic_dataset(
    samples_per_class: int = 200, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic dataset grounded in published ICAR epidemiological rules.

    Features: [temp_min_c, temp_max_c, humidity_3d_avg, rain_7d_mm, soil_moisture, crop_id]
    Classes (0-7):
      0: healthy
      1: late_blight (potato)
      2: early_blight (potato / tomato)
      3: rice_blast (rice)
      4: bacterial_leaf_blight (rice)
      5: yellow_rust (wheat)
      6: powdery_mildew (wheat)
      7: tomato_leaf_curl (tomato)
    """
    np.random.seed(random_state)
    X_list = []
    y_list = []

    for disease_id in range(len(SUPPORTED_DISEASES)):
        n = samples_per_class

        if disease_id == 0:  # healthy (all crops, moderate weather)
            crop = np.random.choice([0, 1, 2, 3], size=n)
            t_min = np.random.normal(18.0, 3.0, size=n)
            t_max = t_min + np.random.normal(10.0, 2.0, size=n)
            hum = np.clip(np.random.normal(55.0, 10.0, size=n), 20.0, 75.0)
            rain = np.clip(np.random.exponential(5.0, size=n), 0.0, 20.0)
            sm = np.clip(np.random.normal(0.20, 0.04, size=n), 0.10, 0.28)

        elif disease_id == 1:  # late_blight (potato: cool nights, very high humidity)
            crop = np.full(n, 0)
            t_min = np.random.normal(12.5, 2.0, size=n)
            t_max = np.random.normal(21.0, 2.5, size=n)
            hum = np.clip(np.random.normal(91.0, 4.0, size=n), 82.0, 99.0)
            rain = np.clip(np.random.normal(35.0, 12.0, size=n), 12.0, 90.0)
            sm = np.clip(np.random.normal(0.28, 0.04, size=n), 0.18, 0.38)

        elif (
            disease_id == 2
        ):  # early_blight (potato / tomato: warm, alternating wet-dry)
            crop = np.random.choice([0, 3], size=n)
            t_min = np.random.normal(20.0, 2.5, size=n)
            t_max = np.random.normal(29.0, 3.0, size=n)
            hum = np.clip(np.random.normal(72.0, 8.0, size=n), 55.0, 85.0)
            rain = np.clip(np.random.exponential(12.0, size=n), 2.0, 45.0)
            sm = np.clip(np.random.normal(0.22, 0.04, size=n), 0.12, 0.30)

        elif disease_id == 3:  # rice_blast (rice: warm, heavy dew, humidity > 90%)
            crop = np.full(n, 1)
            t_min = np.random.normal(22.0, 2.0, size=n)
            t_max = np.random.normal(28.0, 2.5, size=n)
            hum = np.clip(np.random.normal(93.0, 3.5, size=n), 86.0, 99.0)
            rain = np.clip(np.random.normal(45.0, 15.0, size=n), 20.0, 120.0)
            sm = np.clip(np.random.normal(0.32, 0.05, size=n), 0.22, 0.42)

        elif disease_id == 4:  # bacterial_leaf_blight (rice: warm, windy rain, 25-34 C)
            crop = np.full(n, 1)
            t_min = np.random.normal(25.0, 2.0, size=n)
            t_max = np.random.normal(33.0, 2.5, size=n)
            hum = np.clip(np.random.normal(84.0, 6.0, size=n), 72.0, 96.0)
            rain = np.clip(np.random.normal(40.0, 14.0, size=n), 15.0, 100.0)
            sm = np.clip(np.random.normal(0.30, 0.05, size=n), 0.20, 0.40)

        elif disease_id == 5:  # yellow_rust (wheat: cool moist weather, 8-15 C)
            crop = np.full(n, 2)
            t_min = np.random.normal(7.5, 2.0, size=n)
            t_max = np.random.normal(16.0, 2.5, size=n)
            hum = np.clip(np.random.normal(83.0, 5.0, size=n), 72.0, 95.0)
            rain = np.clip(np.random.exponential(10.0, size=n), 1.0, 35.0)
            sm = np.clip(np.random.normal(0.24, 0.04, size=n), 0.14, 0.32)

        elif disease_id == 6:  # powdery_mildew (wheat: moderate temp, 15-22 C, humid)
            crop = np.full(n, 2)
            t_min = np.random.normal(14.0, 2.0, size=n)
            t_max = np.random.normal(23.0, 2.5, size=n)
            hum = np.clip(np.random.normal(76.0, 6.0, size=n), 65.0, 88.0)
            rain = np.clip(np.random.exponential(4.0, size=n), 0.0, 15.0)
            sm = np.clip(np.random.normal(0.21, 0.03, size=n), 0.12, 0.29)

        elif (
            disease_id == 7
        ):  # tomato_leaf_curl (tomato: hot, lower humidity, whitefly)
            crop = np.full(n, 3)
            t_min = np.random.normal(24.0, 2.5, size=n)
            t_max = np.random.normal(35.0, 3.0, size=n)
            hum = np.clip(np.random.normal(48.0, 9.0, size=n), 30.0, 68.0)
            rain = np.clip(np.random.exponential(3.0, size=n), 0.0, 15.0)
            sm = np.clip(np.random.normal(0.18, 0.03, size=n), 0.10, 0.25)

        features = np.column_stack([t_min, t_max, hum, rain, sm, crop])
        X_list.append(features)
        y_list.append(np.full(n, disease_id))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def compute_multiclass_brier(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 8
) -> float:
    """Computes multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2."""
    y_one_hot = label_binarize(y_true, classes=list(range(num_classes)))
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    preds = np.argmax(y_prob, axis=1)
    confs = np.max(y_prob, axis=1)
    accuracies = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def benchmark_cpu_latency_ms(
    model_predict_fn, sample_x: np.ndarray, iterations: int = 100
) -> float:
    """Measures p99 CPU inference latency in milliseconds over N iterations."""
    latencies = []
    # Warmup
    for _ in range(10):
        model_predict_fn(sample_x)

    for _ in range(iterations):
        t0 = time.perf_counter()
        model_predict_fn(sample_x)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return float(np.percentile(latencies, 99))


def run_5fold_ablation_study(
    n_splits: int = 5,
    samples_per_class: int = 200,
    export_json_path: str = "artifacts/benchmark_results.json",
) -> dict[str, Any]:
    """Runs rigorous 5-fold CV ablation study across all 5 model families."""
    print("=== Starting 5-Fold Agronomic ML Model Ablation Benchmark ===")
    X, y = generate_agronomic_dataset(samples_per_class=samples_per_class)
    print(
        f"Dataset Shape: X = {X.shape}, y = {y.shape} across {len(SUPPORTED_DISEASES)} disease classes"
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_names = [
        "LogisticRegression",
        "MLPClassifier",
        "RandomForest",
        "BaselineXGBPredictor",
        "CalibratedTriadPredictor",
    ]

    metrics: dict[str, dict[str, list[float]]] = {
        name: {"f1_macro": [], "roc_auc": [], "brier": [], "ece": []}
        for name in model_names
    }

    sample_x = X[:1]
    latencies: dict[str, float] = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Running Fold {fold}/{n_splits} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        models = {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "MLPClassifier": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=300, random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42
            ),
            "BaselineXGBPredictor": BaselineXGBPredictor(
                n_estimators=100, max_depth=4, learning_rate=0.08
            ),
            "CalibratedTriadPredictor": CalibratedTriadPredictor(
                weights=[0.35, 0.35, 0.30]
            ),
        }

        for name, m in models.items():
            m.fit(X_train, y_train)

            # Predict probabilities
            if hasattr(m, "predict_proba"):
                probs = m.predict_proba(X_test)
            elif hasattr(m, "model") and hasattr(m.model, "predict_proba"):
                probs = m.model.predict_proba(X_test)
            else:
                raise ValueError(f"Model {name} has no predict_proba method")

            preds = np.argmax(probs, axis=1)

            # Metrics
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
            brier = compute_multiclass_brier(
                y_test, probs, num_classes=len(SUPPORTED_DISEASES)
            )
            ece = compute_expected_calibration_error(y_test, probs, num_bins=10)

            metrics[name]["f1_macro"].append(float(f1))
            metrics[name]["roc_auc"].append(float(auc))
            metrics[name]["brier"].append(float(brier))
            metrics[name]["ece"].append(float(ece))

            print(
                f"  [{name:24s}] F1: {f1:.4f} | ROC-AUC: {auc:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}"
            )

            # Latency benchmark on fold 1
            if fold == 1:
                if hasattr(m, "predict_proba"):
                    fn = m.predict_proba
                else:
                    fn = m.model.predict_proba
                lat = benchmark_cpu_latency_ms(fn, sample_x, iterations=100)
                latencies[name] = round(lat, 2)

    # Compute summary statistics
    results: dict[str, Any] = {
        "benchmark_metadata": {
            "num_folds": n_splits,
            "total_samples": len(y),
            "num_classes": len(SUPPORTED_DISEASES),
            "features": [
                "temp_min_c",
                "temp_max_c",
                "humidity_3d_avg",
                "rain_7d_mm",
                "soil_moisture",
                "crop_id",
            ],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "models": {},
    }

    print("\n" + "=" * 90)
    print(
        "=== FINAL 5-FOLD ABLATION BENCHMARK RESULTS (Scoring Rubric Verification) ==="
    )
    print("=" * 90)
    print(
        f"{'Model Architecture':<26} | {'Macro-F1 (Mean±Std)':<18} | {'ROC-AUC':<10} | {'Brier (Lower)':<13} | {'ECE (Lower)':<11} | {'p99 Latency':<10}"
    )
    print("-" * 90)

    for name in model_names:
        f1_mean = float(np.mean(metrics[name]["f1_macro"]))
        f1_std = float(np.std(metrics[name]["f1_macro"]))
        auc_mean = float(np.mean(metrics[name]["roc_auc"]))
        brier_mean = float(np.mean(metrics[name]["brier"]))
        ece_mean = float(np.mean(metrics[name]["ece"]))
        p99_lat = latencies[name]

        results["models"][name] = {
            "f1_macro_mean": round(f1_mean, 4),
            "f1_macro_std": round(f1_std, 4),
            "roc_auc_macro_mean": round(auc_mean, 4),
            "brier_score_mean": round(brier_mean, 4),
            "ece_score_mean": round(ece_mean, 4),
            "p99_cpu_latency_ms": p99_lat,
        }

        print(
            f"{name:<26} | {f1_mean:.4f} ± {f1_std:.4f}    | {auc_mean:.4f}     | {brier_mean:.4f}        | {ece_mean:.4f}      | {p99_lat:.2f} ms"
        )
    print("=" * 90)

    # Export benchmark JSON
    os.makedirs(os.path.dirname(os.path.abspath(export_json_path)), exist_ok=True)
    with open(export_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"\n[AblationSuite] Successfully exported benchmark results to: {export_json_path}"
    )

    # Retrain and persist production models on full dataset
    print("\n[Production] Retraining final production models on full 100% dataset...")
    prod_xgb = BaselineXGBPredictor(n_estimators=100, max_depth=4, learning_rate=0.08)
    prod_xgb.fit(X, y)
    prod_xgb.save("artifacts/xgb_baseline.joblib")
    print("  -> Saved Baseline XGBoost model to artifacts/xgb_baseline.joblib")

    prod_triad = CalibratedTriadPredictor(weights=[0.35, 0.35, 0.30])
    prod_triad.fit(X, y)
    prod_triad.save("artifacts/calibrated_triad.joblib")
    print("  -> Saved Calibrated Triad model to artifacts/calibrated_triad.joblib")

    return results


if __name__ == "__main__":
    run_5fold_ablation_study()
