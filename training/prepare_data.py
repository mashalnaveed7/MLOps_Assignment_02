from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    root = Path(__file__).resolve().parents[1]
    output_path = root / "data" / "raw" / "breast_cancer.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.data.copy()
    df["target"] = dataset.target

    if df.isnull().sum().sum() != 0:
        raise ValueError("Dataset contains missing values.")

    if len(df.columns) - 1 != 30:
        raise ValueError("Dataset must contain exactly 30 features.")

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")


if __name__ == "__main__":
    main()
