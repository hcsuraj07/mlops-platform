# =============================================================================
# train.py
# PURPOSE: Train an XGBoost model on the credit default dataset with full
#          MLflow experiment tracking — every run is logged, versioned, and
#          stored in the model registry.
#
# WHAT IS MLFLOW?
#   MLflow is an open-source platform for managing the ML lifecycle.
#   It tracks:
#     - Parameters  → what settings did you use? (learning rate, max depth)
#     - Metrics     → how did the model perform? (AUC, F1, accuracy)
#     - Artifacts   → the actual model file, plots, feature importance
#     - Tags        → metadata (who ran it, what dataset, git commit)
#
#   WHY DOES THIS MATTER?
#   Without MLflow, you train a model, get 0.84 AUC, tweak something, get
#   0.86 AUC, and have no idea what changed. With MLflow, every experiment
#   is recorded — you can compare runs, reproduce any result, and promote
#   the best model to production with one click.
#
# FLOW:
#   load data → preprocess → split → train XGBoost → evaluate
#   → log everything to MLflow → register best model
# =============================================================================

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# MLFLOW SETUP
#
# mlflow.set_tracking_uri() tells MLflow where to store experiment data.
# "mlruns" is a local folder — every run creates a subfolder here with
# all params, metrics, and model files saved as JSON and artifacts.
#
# mlflow.set_experiment() creates a named experiment to group related runs.
# Think of an experiment as a folder for all your training attempts on
# the same problem. If it doesn't exist, MLflow creates it automatically.
# -----------------------------------------------------------------------------
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("credit-default-prediction")


def load_and_preprocess(data_path: str = "data/raw/credit_default.csv"):
    """
    Load raw data and apply feature engineering.

    WHY RENAME COLUMNS?
    The dataset uses cryptic names (X1-X23). Renaming makes the model
    interpretable — when SHAP shows feature importance, "payment_status_sep"
    means something. "X6" means nothing.

    WHY SCALE FEATURES?
    XGBoost is tree-based and doesn't strictly need scaling, but scaling
    bill amounts (range: -300k to 1M) helps with numerical stability and
    makes feature importance more comparable across features.
    """
    df = pd.read_csv(data_path)

    # rename features to meaningful names
    # X1-X5: demographic info
    # X6-X11: payment status (-1=on time, 1-9=months delayed)
    # X12-X17: bill statement amounts
    # X18-X23: previous payment amounts
    column_map = {
        "X1": "credit_limit",
        "X2": "gender",
        "X3": "education",
        "X4": "marriage",
        "X5": "age",
        "X6": "payment_status_sep",
        "X7": "payment_status_aug",
        "X8": "payment_status_jul",
        "X9": "payment_status_jun",
        "X10": "payment_status_may",
        "X11": "payment_status_apr",
        "X12": "bill_amt_sep",
        "X13": "bill_amt_aug",
        "X14": "bill_amt_jul",
        "X15": "bill_amt_jun",
        "X16": "bill_amt_may",
        "X17": "bill_amt_apr",
        "X18": "payment_amt_sep",
        "X19": "payment_amt_aug",
        "X20": "payment_amt_jul",
        "X21": "payment_amt_jun",
        "X22": "payment_amt_may",
        "X23": "payment_amt_apr",
    }
    df = df.rename(columns=column_map)

    # feature engineering
    df["utilization_ratio"] = df["bill_amt_sep"] / (df["credit_limit"] + 1)

    payment_cols = ["payment_status_sep", "payment_status_aug", "payment_status_jul",
                    "payment_status_jun", "payment_status_may", "payment_status_apr"]
    df["total_delays"] = df[payment_cols].apply(lambda x: (x > 0).sum(), axis=1)

    df["avg_payment_ratio"] = (
        df[["payment_amt_sep", "payment_amt_aug", "payment_amt_jul"]].mean(axis=1) /
        (df[["bill_amt_sep", "bill_amt_aug", "bill_amt_jul"]].mean(axis=1) + 1)
    )

    # clip infinite and very large values caused by division
    # replace inf/-inf with NaN first, then fill with 0
    import numpy as np
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # separate features and target
    X = df.drop("default", axis=1)
    y = df["default"]

    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"Default rate: {y.mean():.2%}")

    return X, y  # ← this line must be there

def train(params: dict = None):
    """
    Full training pipeline with MLflow tracking.

    WHAT GETS LOGGED TO MLFLOW:
      - params: all XGBoost hyperparameters
      - metrics: AUC, F1, accuracy, precision, recall on test set
      - metrics: cross-validation AUC mean and std
      - artifacts: feature importance plot, classification report
      - model: the trained XGBoost model (loadable later for serving)
      - tags: dataset version, model type
    """

    # default hyperparameters — tuned for this dataset
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 3,  # handles class imbalance (22% default rate)
            "random_state": 42,
            "eval_metric": "auc",
            "use_label_encoder": False,
        }

    # load and preprocess data
    X, y = load_and_preprocess()

    # train/test split — stratified to preserve default rate in both sets
    # WHY STRATIFIED? With 22% defaults, a random split might give you
    # 18% in one set and 26% in another. Stratified ensures both sets
    # have ~22% defaults, making evaluation reliable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ---------------------------------------------------------------------
    # START MLFLOW RUN
    # Everything inside this block is tracked automatically.
    # mlflow.start_run() creates a new run with a unique ID.
    # ---------------------------------------------------------------------
    with mlflow.start_run(run_name="xgboost-v1") as run:
        run_id = run.info.run_id
        print(f"\nMLflow run ID: {run_id}")

        # LOG PARAMETERS — what settings are we using?
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("dataset", "UCI Credit Default")
        mlflow.set_tag("dataset_size", len(X))

        # TRAIN THE MODEL
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # EVALUATE ON TEST SET
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # calculate metrics
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # LOG METRICS — how did the model perform?
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)

        print(f"\nTest Results:")
        print(f"  AUC:       {auc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")

        # CROSS VALIDATION — more reliable than single train/test split
        # WHY CV? A single split can be lucky or unlucky. 5-fold CV trains
        # on 5 different splits and averages — gives a truer picture of
        # model quality and variance.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_scores.std())
        print(f"\n  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # LOG FEATURE IMPORTANCE PLOT as an artifact
        os.makedirs("notebooks", exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=True).tail(15)

        importance_df.plot(kind="barh", x="feature", y="importance", ax=ax)
        ax.set_title("Top 15 Feature Importances")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("notebooks/feature_importance.png", dpi=100)
        plt.close()

        # mlflow.log_artifact uploads the plot file to the run
        mlflow.log_artifact("notebooks/feature_importance.png")

        # LOG THE MODEL to MLflow model registry
        # This saves the model in a standard format that can be loaded
        # later with mlflow.xgboost.load_model() for serving
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="credit-default-xgboost"
        )

        print(f"\nModel registered as: credit-default-xgboost")
        print(f"MLflow UI: run `mlflow ui` to view this run")

        return run_id, auc


if __name__ == "__main__":
    run_id, auc = train()
    print(f"\nDone. Run ID: {run_id} | AUC: {auc:.4f}")