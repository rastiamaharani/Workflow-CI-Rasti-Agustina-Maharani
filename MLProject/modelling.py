import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV

warnings.filterwarnings("ignore")


def _find_preprocessed_dir(base_dir: Path) -> Path:
    env_dir = os.getenv("PREPROCESSED_DIR")
    if env_dir:
        p = (base_dir / env_dir).resolve() if not Path(env_dir).is_absolute() else Path(env_dir)
        if (p / "train_preprocessed.csv").exists() and (p / "test_preprocessed.csv").exists():
            return p

    for child in base_dir.iterdir():
        if child.is_dir():
            if (child / "train_preprocessed.csv").exists() and (child / "test_preprocessed.csv").exists():
                return child

    raise FileNotFoundError(
        "Tidak menemukan folder preprocessing. Pastikan ada:\n"
        "MLProject/<namadataset_preprocessing>/train_preprocessed.csv dan test_preprocessed.csv"
    )


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    # Penting untuk CI build-docker:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_scoring_ci")
    mlflow.set_experiment(exp_name)

    # Load data
    pre_dir = _find_preprocessed_dir(base_dir)
    train_df = pd.read_csv(pre_dir / "train_preprocessed.csv")
    test_df = pd.read_csv(pre_dir / "test_preprocessed.csv")

    target_col = os.getenv("TARGET_COL", "Y")
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Kolom target '{target_col}' tidak ada di file preprocessed.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    # Model
    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    do_tuning = os.getenv("DO_TUNING", "1") == "1"
    best_model = base_model
    best_params = {}

    if do_tuning:
        param_grid = {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        gs = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1",
            n_jobs=-1,
            cv=cv,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        best_model.fit(X_train, y_train)

    # Eval
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    cm = confusion_matrix(y_test, y_pred)

    # PENTING: Pakai run_id dari MLFLOW_RUN_ID jika ada 
    run_id_env = os.environ.get("MLFLOW_RUN_ID")
    run_name = os.getenv("RUN_NAME", "ci_retraining_rf")

    with mlflow.start_run(run_id=run_id_env, run_name=None if run_id_env else run_name) as run:
        # tags
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset_dir", str(pre_dir.name))
        mlflow.set_tag("target_col", target_col)

        # params
        mlflow.log_param("do_tuning", int(do_tuning))
        mlflow.log_param("n_estimators", getattr(best_model, "n_estimators", None))
        mlflow.log_param("class_weight", getattr(best_model, "class_weight", None))
        if best_params:
            for k, v in best_params.items():
                mlflow.log_param(f"best_{k}", v)

        # metrics
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        if not np.isnan(roc):
            mlflow.log_metric("roc_auc", float(roc))

        # artifact: confusion_matrix
        cm_txt = "\n".join(["\t".join(map(str, row)) for row in cm.tolist()])
        cm_path = base_dir / "confusion_matrix.txt"
        cm_path.write_text(
            "Confusion Matrix (rows=true, cols=pred):\n" + cm_txt,
            encoding="utf-8",
        )
        mlflow.log_artifact(str(cm_path))
        try:
            cm_path.unlink()
        except Exception:
            pass

        # log model HARUS artifact_path="model"
        input_example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
        )

        # TULIS RUN_ID untuk step build-docker CI
        (base_dir / "last_run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        print("[OK] Training + logging selesai.")
        print(f"[OK] RUN_ID: {run.info.run_id}")
        print(f"[OK] Metrics: acc={acc:.4f}, f1={f1:.4f}, prec={prec:.4f}, rec={rec:.4f}, roc_auc={roc:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())