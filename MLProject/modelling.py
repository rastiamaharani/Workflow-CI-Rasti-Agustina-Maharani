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


# Utilities
def _find_preprocessed_dir(base_dir: Path) -> Path:

    env_dir = os.getenv("PREPROCESSED_DIR")
    if env_dir:
        p = (base_dir / env_dir).resolve() if not Path(env_dir).is_absolute() else Path(env_dir)
        if (p / "train_preprocessed.csv").exists() and (p / "test_preprocessed.csv").exists():
            return p

    # scan subfolder
    for child in base_dir.iterdir():
        if child.is_dir():
            if (child / "train_preprocessed.csv").exists() and (child / "test_preprocessed.csv").exists():
                return child

    raise FileNotFoundError(
        "Tidak menemukan folder preprocessing. Pastikan ada:\n"
        "MLProject/<namadataset_preprocessing>/train_preprocessed.csv dan test_preprocessed.csv"
    )


def _setup_tracking():
    if os.getenv("MLFLOW_TRACKING_URI"):
        return

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER") or os.getenv("REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME") or os.getenv("REPO_NAME")

    # DagsHub init 
    try:
        import dagshub  # noqa

        if repo_owner and repo_name:
            # dagshub.init akan set tracking ke DagsHub MLflow endpoint
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            print(f"[INFO] DagsHub tracking aktif: {repo_owner}/{repo_name}")
        else:
            print("[INFO] DAGSHUB_REPO_OWNER/NAME tidak diset, pakai local tracking MLflow.")
    except Exception as e:
        print(f"[WARN] DagsHub init gagal, fallback local MLflow. Detail: {e}")


def _safe_start_run(run_name: str):

    active = mlflow.active_run()
    if active is not None:
        print(f"[INFO] Active run terdeteksi: {active.info.run_id} (gunakan run ini)")
        return None  # menandakan "tidak membuka run baru"
    return mlflow.start_run(run_name=run_name)


# Main
def main():
    base_dir = Path(__file__).resolve().parent

    # 1) Setup tracking 
    _setup_tracking()

    # 2) Set experiment name 
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_scoring_exp")
    mlflow.set_experiment(exp_name)

    # 3) Load dataset preprocessed
    pre_dir = _find_preprocessed_dir(base_dir)
    train_path = pre_dir / "train_preprocessed.csv"
    test_path = pre_dir / "test_preprocessed.csv"

    target_col = os.getenv("TARGET_COL", "Y")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Kolom target '{target_col}' tidak ada di file preprocessed.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    # 4) Training config
    # Default aman: RandomForest dengan class_weight untuk imbalance
    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Optional: GridSearch 
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

    # 5) Evaluasi
    y_pred = best_model.predict(X_test)
    # Untuk ROC-AUC butuh probas
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

    cm = confusion_matrix(y_test, y_pred)

    # 6) Logging ke MLflow (tanpa bikin error run)
    run_ctx = _safe_start_run(run_name=os.getenv("RUN_NAME", "ci_retraining_rf"))

    try:
        # tags/info penting
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

        # confusion matrix sebagai artifact txt 
        cm_txt = "\n".join(["\t".join(map(str, row)) for row in cm.tolist()])
        cm_path = base_dir / "confusion_matrix.txt"
        cm_path.write_text(
            "Confusion Matrix (rows=true, cols=pred):\n" + cm_txt,
            encoding="utf-8",
        )
        mlflow.log_artifact(str(cm_path))
        try:
            cm_path.unlink(missing_ok=True)
        except Exception:
            pass

        # log model -> artifact name HARUS "model" 
        input_example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
        )

        print("[OK] Training + logging selesai.")
        print(f"[OK] Metrics: acc={acc:.4f}, f1={f1:.4f}, prec={prec:.4f}, rec={rec:.4f}, roc_auc={roc:.4f}")

    finally:
        if run_ctx is not None:
            run_ctx.__exit__(None, None, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())