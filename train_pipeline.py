"""
Module 2: Hybrid ML Training Pipeline (train_pipeline.py)
Trains Engine A (XGBoost Supervised Multi-class) and Engine B (Isolation Forest Unsupervised Zero-Day Detector).
Exports models, scalers, and evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

from data_generator import (
    FEATURE_NAMES,
    KNOWN_CLASSES,
    generate_training_data,
    export_sample_test_flows,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def run_training_pipeline():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("=" * 60)
    print("🚀 Initializing Hybrid ML Training Pipeline...")
    print("=" * 60)

    # 1. Generate Training Data
    print("[1/5] Synthesizing CIC-IDS2017 balanced telemetry (6,000 samples)...")
    df = generate_training_data(n_samples=6000)
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    # 2. Stratified 80/20 Split & Preprocessing
    print("[2/5] Executing 80/20 Stratified train/val split & feature scaling...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Engine A: Supervised Multi-Class Classifier
    print("[3/5] Training Engine A (XGBClassifier: 150 estimators, max_depth=6)...")
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    xgb_model.fit(X_train_scaled, y_train_enc)

    # 4. Engine B: Unsupervised Zero-Day Anomaly Detector
    print("[4/5] Training Engine B (IsolationForest: strictly on Normal baseline flows)...")
    normal_label_enc = label_encoder.transform(["Normal"])[0]
    normal_mask = (y_train_enc == normal_label_enc)
    X_train_normal = X_train_scaled[normal_mask]

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42,
    )
    iso_forest.fit(X_train_normal)

    # 5. Model Evaluation
    print("[5/5] Evaluating performance and calculating comprehensive metrics...")
    y_pred = xgb_model.predict(X_val_scaled)
    y_proba = xgb_model.predict_proba(X_val_scaled)

    acc = float(accuracy_score(y_val_enc, y_pred))
    macro_prec = float(precision_score(y_val_enc, y_pred, average="macro"))
    macro_rec = float(recall_score(y_val_enc, y_pred, average="macro"))
    macro_f1 = float(f1_score(y_val_enc, y_pred, average="macro"))
    roc_auc = float(roc_auc_score(y_val_enc, y_proba, multi_class="ovr", average="weighted"))

    cm = confusion_matrix(y_val_enc, y_pred).tolist()
    class_labels = label_encoder.classes_.tolist()

    report_dict = classification_report(y_val_enc, y_pred, target_names=class_labels, output_dict=True)
    per_class_metrics = {}
    for cls in class_labels:
        per_class_metrics[cls] = {
            "precision": round(float(report_dict[cls]["precision"]), 4),
            "recall": round(float(report_dict[cls]["recall"]), 4),
            "f1_score": round(float(report_dict[cls]["f1-score"]), 4),
            "support": int(report_dict[cls]["support"]),
        }

    # Global feature importances
    importances = xgb_model.feature_importances_
    global_feature_importances = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    ]

    metrics_payload = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_roc_auc": round(roc_auc, 4),
        "class_labels": class_labels,
        "confusion_matrix": cm,
        "per_class": per_class_metrics,
        "global_feature_importances": global_feature_importances,
    }

    # Save artifacts
    joblib.dump(xgb_model, os.path.join(ARTIFACTS_DIR, "xgb_model.joblib"))
    joblib.dump(iso_forest, os.path.join(ARTIFACTS_DIR, "iso_forest.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))

    metrics_path = os.path.join(ARTIFACTS_DIR, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Export sample test flows CSV
    sample_csv_path = os.path.join(ARTIFACTS_DIR, "sample_test_flows.csv")
    export_sample_test_flows(sample_csv_path)

    print("=" * 60)
    print(f"✅ Training Pipeline Completed Successfully!")
    print(f"   Accuracy:         {acc * 100:.2f}%")
    print(f"   Macro Precision:  {macro_prec * 100:.2f}%")
    print(f"   Macro Recall:     {macro_rec * 100:.2f}%")
    print(f"   Macro F1:         {macro_f1 * 100:.2f}%  (Requirement: > 90%)")
    print(f"   Weighted ROC-AUC: {roc_auc:.4f}")
    print(f"   Artifacts saved to: {ARTIFACTS_DIR}")
    print("=" * 60)

    return metrics_payload


if __name__ == "__main__":
    run_training_pipeline()
