"""
Generates a minimal synthetic churn model artifact for CI use only.
Produces the same artifact structure as scripts/export_model.py but
uses a tiny synthetic dataset — no real CSV required.

DO NOT USE IN PRODUCTION. For CI Docker build verification only.
"""
import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

FEATURE_COUNT = 51
OUTPUT_PATH = "models/churn_model.joblib"


def generate_ci_artifact(output_path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Tiny synthetic dataset — just enough to fit the model
    rng = np.random.default_rng(42)
    X = rng.random((200, FEATURE_COUNT))
    y = rng.integers(0, 2, size=200)

    model = RandomForestClassifier(
        n_estimators=10,          # minimal — fast to train in CI
        random_state=42,
        n_jobs=1
    )
    model.fit(X, y)

    artifact = {
        "model":          model,
        "feature_count":  FEATURE_COUNT,
        "model_version":  "ci-synthetic",
        "trained_on":     datetime.utcnow().isoformat(),
        "dataset_rows":   200,
        "train_rows":     160,
        "test_rows":      40,
        "roc_auc":        0.5,    # synthetic — not meaningful
        "accuracy":       0.5,    # synthetic — not meaningful
    }

    joblib.dump(artifact, output_path, compress=3)
    print(f"CI model artifact written to {output_path}")
    print(f"  feature_count: {FEATURE_COUNT}")
    print(f"  model_version: ci-synthetic")


if __name__ == "__main__":
    generate_ci_artifact()
