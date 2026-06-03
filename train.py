"""
Simple training script:
- pulls Iris.csv from S3 via DVC (versioned)
- trains a LogisticRegression
- saves model to artifacts/
"""

import os
import json
import subprocess
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Data config
DATA_PATH = "data/Iris.csv"

def pull_data():
    """Pull the correct versioned data from S3 via DVC."""
    print("Pulling data from S3 via DVC...")
    result = subprocess.run(
        ["dvc", "pull", DATA_PATH],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"DVC pull failed:\n{result.stderr}")
    print(f"DVC pull successful: {result.stdout.strip()}")

def load_data():
    """Load Iris.csv from local (after dvc pull)."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    return df

def main():
    # Step 1: Pull correct data version from S3
    pull_data()

    # Step 2: Load into dataframe
    df = load_data()

    # Step 3: Preprocess
    df = df.drop(columns=["Id"])

    le = LabelEncoder()
    df["Species"] = le.fit_transform(df["Species"])
    # "Iris-setosa"     → 0
    # "Iris-versicolor" → 1
    # "Iris-virginica"  → 2

    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    # Step 4: Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Step 5: Save model and encoder
    os.makedirs("artifacts", exist_ok=True)

    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)

    encoder_path = os.path.join("artifacts", "label_encoder.pkl")
    joblib.dump(le, encoder_path)

    # Step 6: Save metrics
    acc = model.score(X_test, y_test)
    metrics = {
        "accuracy": float(acc),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "data_version": open("data/Iris.csv.dvc").read()  # log which version was used
    }
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved label encoder to {encoder_path}")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()