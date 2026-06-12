import os
import joblib
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd

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


def evaluate_model(model, X_test, y_test):
    """
    Evaluasi model pada test set.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "roc_auc": roc_auc,
    }

    return metrics, y_pred

def main():
    RANDOM_STATE = 42

    TARGET_COLUMN = "num"
    DATA_DIR = "./"

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
        "C": 3.1577304105435946,
        "gamma": "auto",
        "class_weight": None,
    }

    init_dagshub(
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
    )

    with mlflow.start_run(run_name="heart_disease_svc"):
        model.fit(X_train, y_train)

        metrics, y_pred = evaluate_model(model, X_test, y_test)

        print("\nTest Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        os.makedirs("models", exist_ok=True)
        model_path = "models/heart_disease_svc.joblib"
        joblib.dump(model, model_path)

        print(f"\nModel saved locally to: {model_path}")
        print("Training dan logging MLflow ke DagsHub selesai.")


if __name__ == "__main__":
    main()
