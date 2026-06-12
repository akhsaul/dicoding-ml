import os
import json
import joblib
import optuna
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def init_dagshub(
    repo_owner: str,
    repo_name: str,
    experiment_name: str,
    token: str | None = None,
    autolog: bool = False,
):
    if token:
        dagshub.auth.add_app_token(token)

    dagshub.init(
        repo_owner=repo_owner,
        repo_name=repo_name,
        mlflow=True,
    )

    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_models=autolog)


def load_train_test_data(
    x_train_path: str,
    x_test_path: str,
    y_train_path: str,
    y_test_path: str,
    target_column: str,
):
    X_train = pd.read_csv(x_train_path).astype("float64")
    X_test = pd.read_csv(x_test_path).astype("float64")

    y_train = pd.read_csv(y_train_path)[target_column].astype(int)
    y_test = pd.read_csv(y_test_path)[target_column].astype(int)

    return X_train, X_test, y_train, y_test


def suggest_svc_params(trial):
    kernel = trial.suggest_categorical(
        "kernel",
        ["rbf", "linear", "poly", "sigmoid"],
    )

    params = {
        "kernel": kernel,
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }

    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
        params["coef0"] = trial.suggest_float("coef0", 0.0, 2.0)

    if kernel == "sigmoid":
        params["coef0"] = trial.suggest_float("coef0", 0.0, 2.0)

    return params


def build_model(
    params: dict,
    random_state: int,
):
    svc_params = {
        "kernel": params["kernel"],
        "C": params["C"],
        "gamma": params["gamma"],
        "class_weight": params["class_weight"],
        "random_state": random_state,
    }

    if params["kernel"] == "poly":
        svc_params["degree"] = params.get("degree", 3)
        svc_params["coef0"] = params.get("coef0", 0.0)

    if params["kernel"] == "sigmoid":
        svc_params["coef0"] = params.get("coef0", 0.0)

    base_svc = SVC(**svc_params)

    model = CalibratedClassifierCV(
        estimator=base_svc,
        method="sigmoid",
        cv=5,
        ensemble=False,
    )

    return model


def objective(
    trial,
    X_train,
    y_train,
    random_state: int,
):
    params = suggest_svc_params(trial)

    model = build_model(
        params=params,
        random_state=random_state,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    with mlflow.start_run(
        run_name=f"heart_disease_svc_optuna_trial_{trial.number}",
        nested=True,
    ):
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_params(params)

        scores = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )

        mean_f1 = scores.mean()
        std_f1 = scores.std()

        mlflow.log_metric("cv_f1_mean", mean_f1)
        mlflow.log_metric("cv_f1_std", std_f1)

        for fold_idx, fold_score in enumerate(scores, start=1):
            mlflow.log_metric(f"cv_f1_fold_{fold_idx}", fold_score)

        return mean_f1


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "roc_auc": roc_auc_score(
            y_true=y_test,
            y_score=y_proba,
        ),
    }

    return metrics, y_pred, y_proba


def save_json(data, output_path: str):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def save_classification_report(
    y_test,
    y_pred,
    output_path: str,
):
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    save_json(report, output_path)
    return report


def save_confusion_matrix(
    y_test,
    y_pred,
    output_path: str,
):
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
    )

    ax.set_title("Confusion Matrix - Tuned SVC")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_optuna_history_plot(
    study,
    output_path: str,
):
    finished_trials = [trial for trial in study.trials if trial.value is not None]

    trial_numbers = [trial.number for trial in finished_trials]
    trial_values = [trial.value for trial in finished_trials]

    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, trial_values, marker="o")
    plt.xlabel("Trial")
    plt.ylabel("CV F1 Score")
    plt.title("Optuna Optimization History")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_study(
    study_name: str,
    storage: str,
    random_state: int,
):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    return study


def run_training(
    random_state: int,
    target_column: str,
    x_train_path: str,
    x_test_path: str,
    y_train_path: str,
    y_test_path: str,
    experiment_name: str,
    dagshub_repo_owner: str,
    dagshub_repo_name: str,
    dagshub_token: str | None,
    optuna_db_path: str,
    optuna_storage: str,
    optuna_study_name: str,
    n_trials: int,
    artifact_dir: str,
    model_path: str,
    metrics_path: str,
    best_params_path: str,
    classification_report_path: str,
    confusion_matrix_path: str,
    optuna_history_path: str,
    study_summary_path: str,
):
    init_dagshub(
        repo_owner=dagshub_repo_owner,
        repo_name=dagshub_repo_name,
        experiment_name=experiment_name,
        token=dagshub_token,
        autolog=False,
    )

    os.makedirs(artifact_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_train_test_data(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        target_column=target_column,
    )

    print("Data berhasil dimuat.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    study = create_study(
        study_name=optuna_study_name,
        storage=optuna_storage,
        random_state=random_state,
    )

    with mlflow.start_run(run_name="heart_disease_svc_optuna"):
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("optimization_metric", "f1")
        mlflow.log_param("n_trials_requested", n_trials)
        mlflow.log_param("optuna_storage", optuna_storage)
        mlflow.log_param("optuna_study_name", optuna_study_name)
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("total_features", X_train.shape[1])
        mlflow.log_param("random_state", random_state)

        study.optimize(
            lambda trial: objective(
                trial=trial,
                X_train=X_train,
                y_train=y_train,
                random_state=random_state,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1,
            gc_after_trial=True,
        )

        best_params = study.best_params.copy()

        best_model = build_model(
            params=best_params,
            random_state=random_state,
        )

        best_model.fit(X_train, y_train)

        metrics, y_pred, y_proba = evaluate_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
        )

        report = save_classification_report(
            y_test=y_test,
            y_pred=y_pred,
            output_path=classification_report_path,
        )

        save_confusion_matrix(
            y_test=y_test,
            y_pred=y_pred,
            output_path=confusion_matrix_path,
        )

        save_optuna_history_plot(
            study=study,
            output_path=optuna_history_path,
        )

        study_summary = {
            "study_name": optuna_study_name,
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": best_params,
            "n_trials": len(study.trials),
        }

        save_json(metrics, metrics_path)
        save_json(best_params, best_params_path)
        save_json(study_summary, study_summary_path)

        joblib.dump(best_model, model_path)

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1", study.best_value)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_test,
        )

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(best_params_path)
        mlflow.log_artifact(classification_report_path)
        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(optuna_history_path)
        mlflow.log_artifact(study_summary_path)

        if os.path.exists(optuna_db_path):
            mlflow.log_artifact(optuna_db_path)

        print("\nOptuna tuning selesai.")
        print(f"Optuna study lokal tersimpan di: {optuna_db_path}")
        print(f"Best CV F1 Score: {study.best_value}")
        print(f"Best Params: {study.best_params}")

        print("\nTest Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print(f"\nModel lokal disimpan di: {model_path}")
        print("Manual logging MLflow ke DagsHub selesai.")


if __name__ == "__main__":
    RANDOM_STATE = 42

    TARGET_COLUMN = "num"
    DATA_DIR = "./"

    X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
    X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
    Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
    Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")

    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "heart-disease-svc-optuna")

    DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "akhsaul")
    DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "dicoding-MSML")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN", None)

    OPTUNA_DB_PATH = "optuna_studies.db"
    OPTUNA_STORAGE = f"sqlite:///{OPTUNA_DB_PATH}"
    OPTUNA_STUDY_NAME = "heart_disease_optuna_study"

    N_TRIALS = int(os.getenv("MLFLOW_N_TRIALS", "50"))

    ARTIFACT_DIR = "artifacts"
    MODEL_PATH = os.path.join(ARTIFACT_DIR, "heart_disease_svc_optuna.joblib")
    METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")
    BEST_PARAMS_PATH = os.path.join(ARTIFACT_DIR, "best_params.json")
    CLASSIFICATION_REPORT_PATH = os.path.join(
        ARTIFACT_DIR, "classification_report.json"
    )
    CONFUSION_MATRIX_PATH = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    OPTUNA_HISTORY_PATH = os.path.join(ARTIFACT_DIR, "optuna_history.png")
    STUDY_SUMMARY_PATH = os.path.join(ARTIFACT_DIR, "study_summary.json")

    run_training(
        random_state=RANDOM_STATE,
        target_column=TARGET_COLUMN,
        x_train_path=X_TRAIN_PATH,
        x_test_path=X_TEST_PATH,
        y_train_path=Y_TRAIN_PATH,
        y_test_path=Y_TEST_PATH,
        experiment_name=EXPERIMENT_NAME,
        dagshub_repo_owner=DAGSHUB_REPO_OWNER,
        dagshub_repo_name=DAGSHUB_REPO_NAME,
        dagshub_token=DAGSHUB_TOKEN,
        optuna_db_path=OPTUNA_DB_PATH,
        optuna_storage=OPTUNA_STORAGE,
        optuna_study_name=OPTUNA_STUDY_NAME,
        n_trials=N_TRIALS,
        artifact_dir=ARTIFACT_DIR,
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        best_params_path=BEST_PARAMS_PATH,
        classification_report_path=CLASSIFICATION_REPORT_PATH,
        confusion_matrix_path=CONFUSION_MATRIX_PATH,
        optuna_history_path=OPTUNA_HISTORY_PATH,
        study_summary_path=STUDY_SUMMARY_PATH,
    )
