# Production-ready tabular classification pipeline (pandas + scikit-learn).
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
MODEL_DIR = Path("./agent_workspace/trained_models")
MODEL_FILE = MODEL_DIR / "model.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"

TARGET_NAME_CANDIDATES = (
    "target",
    "label",
    "class",
    "classes",
    "y",
    "outcome",
    "quality",
    "category",
    "response",
)


def _discover_csv_files() -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen and resolved.is_file() and resolved.suffix.lower() == ".csv":
            seen.add(key)
            found.append(resolved)

    env_path = os.environ.get("AUTOML_DATA_PATH") or os.environ.get("DATA_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            add(p)
        elif p.is_dir():
            for csv_file in sorted(p.glob("*.csv")):
                add(csv_file)

    for directory in (
        Path("agent_workspace/datasets"),
        Path("./agent_workspace/datasets"),
        Path("."),
    ):
        if directory.is_dir():
            for csv_file in sorted(directory.glob("*.csv")):
                add(csv_file)

    return found


def find_csv_path() -> Path:
    candidates = _discover_csv_files()
    if not candidates:
        raise FileNotFoundError(
            "No CSV dataset found. Set AUTOML_DATA_PATH to a .csv file or directory, "
            "or place a .csv in agent_workspace/datasets/."
        )
    if len(candidates) == 1:
        return candidates[0]

    workspace = [
        p
        for p in candidates
        if "agent_workspace" in str(p).lower() and "datasets" in str(p).lower()
    ]
    if len(workspace) == 1:
        return workspace[0]

    raise FileNotFoundError(
        "Multiple CSV files found; set AUTOML_DATA_PATH to the intended file. "
        f"Candidates: {[str(p) for p in candidates]}"
    )


def infer_target_column(df: pd.DataFrame) -> str:
    lowered = {col: col.lower().strip() for col in df.columns}

    for candidate in TARGET_NAME_CANDIDATES:
        for col, name in lowered.items():
            if (
                name == candidate
                or name.endswith(f"_{candidate}")
                or name.startswith(f"{candidate}_")
            ):
                return col

    categorical_cols: list[tuple[str, int]] = []
    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        if nunique < 2:
            continue
        if series.dtype == "object" or str(series.dtype) == "category":
            if nunique <= max(50, int(len(df) * 0.5)):
                categorical_cols.append((col, nunique))

    if categorical_cols:
        categorical_cols.sort(key=lambda item: item[1])
        return categorical_cols[0][0]

    return df.columns[-1]


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in feature_frame.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found in the dataset.")

    return ColumnTransformer(transformers=transformers)


def main() -> dict:
    csv_path = find_csv_path()
    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    target_column = infer_target_column(dataframe)
    feature_columns = [col for col in dataframe.columns if col != target_column]
    if not feature_columns:
        raise ValueError("No feature columns remain after selecting the target column.")

    features = dataframe[feature_columns]
    target = dataframe[target_column]

    n_classes = int(target.nunique(dropna=True))
    n_features = len(feature_columns)
    if n_classes < 2:
        raise ValueError(
            f"Target column '{target_column}' has fewer than 2 classes; "
            "classification requires at least 2 classes."
        )

    print(f"Dataset: {csv_path}")
    print(f"Target column: {target_column}")
    print(f"Feature columns ({n_features}): {feature_columns}")
    print(f"Number of classes: {n_classes}")
    print(f"Number of input features: {n_features}")
    print(f"Samples: {len(dataframe)}")

    stratify = target if n_classes >= 2 and n_classes < len(target) else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(x_train)
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "dataset_path": str(csv_path),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_samples": int(len(dataframe)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    with open(METRICS_FILE, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        "Model Performance on Test Set:",
        {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
    )
    print("Confusion Matrix:", metrics["confusion_matrix"])
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")

    return metrics


if __name__ == "__main__":
    main()
