import os
import joblib
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
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


def init_tracking(
    repo_owner: str,
    repo_name: str,
    experiment_name: str,
    token: str | None = None,
    autolog: bool = False,
):
    use_dagshub = bool(os.getenv("MLFLOW_TRACKING_URI")) and bool(
        os.getenv("MLFLOW_TRACKING_USERNAME")
    )

    if use_dagshub:
        if token:
            dagshub.auth.add_app_token(token)

        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True,
        )
    else:
        os.environ.pop("MLFLOW_TRACKING_URI", None)

    if not os.getenv("MLFLOW_RUN_ID"):
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

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


def evaluate_model(model, X_test, y_test):
    """
    Evaluasi model pada test set.
    """

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

    return metrics, y_pred


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

    ax.set_title("Confusion Matrix - SVC")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


RANDOM_STATE = 42
TARGET_COLUMN = "num"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "heartdisease_preprocessing")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "heart-disease-svc")

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "akhsaul")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "dicoding-MSML")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN", None)

SVC_BEST_PARAMS = {
    "kernel": "rbf",
    "C": 0.6297705300545552,
    "gamma": "scale",
    "class_weight": None,
    "random_state": RANDOM_STATE,
}


def main():

    init_tracking(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        experiment_name=EXPERIMENT_NAME,
        token=DAGSHUB_TOKEN,
        autolog=True,
    )

    X_train, X_test, y_train, y_test = load_train_test_data(
        x_train_path=X_TRAIN_PATH,
        x_test_path=X_TEST_PATH,
        y_train_path=Y_TRAIN_PATH,
        y_test_path=Y_TEST_PATH,
        target_column=TARGET_COLUMN,
    )

    print("Data berhasil dimuat.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    base_svc = SVC(**SVC_BEST_PARAMS)

    model = CalibratedClassifierCV(
        estimator=base_svc,
        method="sigmoid",
        cv=5,
        ensemble=False,
        n_jobs=-1,
    )

    parent_run_id = os.getenv("MLFLOW_RUN_ID")
    if parent_run_id:
        run_context = mlflow.start_run(run_id=parent_run_id)
    else:
        run_context = mlflow.start_run(run_name="heart_disease_svc")

    with run_context as run:
        model.fit(X_train, y_train)

        metrics, y_pred = evaluate_model(model, X_test, y_test)

        print("\nTest Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        os.makedirs("artifacts", exist_ok=True)
        save_confusion_matrix(
            y_test, y_pred, output_path="artifacts/confusion_matrix.png"
        )

        os.makedirs("artifacts/model", exist_ok=True)
        model_path = "artifacts/model/heart_disease_svc.joblib"
        joblib.dump(model, model_path)
        print(f"\nModel saved locally to: {model_path}")

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"MLflow run_id: {run_id}")

        print("Training dan logging MLflow ke DagsHub selesai.")


if __name__ == "__main__":
    main()
