# Model training: tune an XGBoost model, record every experiment in MLflow,
# check how well the best model does, and save it for the Streamlit app.
import json
import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# In GitHub Actions the workflow starts an MLflow server and tells us its
# address; when we run this by hand, the runs are saved in a local mlruns folder
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-wellness-package-prediction")

# Load the training and test sets that prep.py created
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv")["ProdTaken"]
ytest = pd.read_csv("ytest.csv")["ProdTaken"]

# Text columns get turned into 0/1 columns the model can read; number columns
# get rescaled so that large values (like income) don't dominate
text_columns = ["Occupation", "Gender", "MaritalStatus", "Designation"]
number_columns = [col for col in Xtrain.columns if col not in text_columns]
preprocessor = make_column_transformer(
    (StandardScaler(), number_columns),
    (OneHotEncoder(handle_unknown="ignore"), text_columns),
)

# Only about 1 in 5 customers buys, so we tell XGBoost to pay roughly 4x more
# attention to buyers; otherwise it would learn to just say "no" to everyone
buyer_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
model = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(scale_pos_weight=buyer_weight, random_state=42, eval_metric="logloss"),
)

# The settings we want to try; grid search tests every combination
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],      # number of trees
    "xgbclassifier__max_depth": [3, 5, 7],          # how deep each tree can grow
    "xgbclassifier__learning_rate": [0.05, 0.1],    # how big a step each tree takes
    "xgbclassifier__colsample_bytree": [0.6, 0.8],  # share of columns each tree sees
    "xgbclassifier__subsample": [0.8, 1.0],         # share of rows each tree sees
    "xgbclassifier__reg_lambda": [1, 5],            # penalty that keeps trees simple
}

# We call someone a likely buyer at 45% probability rather than 50%:
# missing a real buyer costs more than making one extra sales call
THRESHOLD = 0.45


def score(X, y):
    """Measure how well the best model predicts buyers in a dataset."""
    probability = best_model.predict_proba(X)[:, 1]
    predicted = (probability >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted),  # of those we flag, how many buy
        "recall": recall_score(y, predicted),        # of real buyers, how many we catch
        "f1": f1_score(y, predicted),                # balance of precision and recall
        "roc_auc": roc_auc_score(y, probability),
    }


with mlflow.start_run(run_name="xgb-gridsearch") as run:
    # Try every combination, using 5-fold cross-validation, and pick the one
    # with the best F1 score
    search = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(Xtrain, ytrain)

    # Record every combination we tried as its own small run in MLflow
    for i, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"combo-{i:02d}", nested=True):
            mlflow.log_params({name.replace("xgbclassifier__", ""): value for name, value in params.items()})
            mlflow.log_metric("cv_mean_f1", search.cv_results_["mean_test_score"][i])
            mlflow.log_metric("cv_std_f1", search.cv_results_["std_test_score"][i])

    # Score the winning model on the training data and on the unseen test data
    best_model = search.best_estimator_
    best_params = {name.replace("xgbclassifier__", ""): value for name, value in search.best_params_.items()}
    metrics = {f"train_{name}": value for name, value in score(Xtrain, ytrain).items()}
    metrics.update({f"test_{name}": value for name, value in score(Xtest, ytest).items()})

    # Record the winner's settings and scores on the main run
    mlflow.log_params(best_params)
    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_metrics(metrics)

    print("Best settings:", best_params)
    for name, value in metrics.items():
        print(f"{name:<16} {value:.4f}")

    # Save the model where the workflow commits it to the repo, plus a small
    # JSON file with the threshold and scores that the app and notebook read
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_model, "tourism_project/deployment/best_tourism_model_v1.joblib")
    with open("tourism_project/deployment/model_metadata.json", "w") as f:
        json.dump({
            "classification_threshold": THRESHOLD,
            "best_params": best_params,
            "metrics": {name: round(float(value), 4) for name, value in metrics.items()},
            "mlflow_run_id": run.info.run_id,
        }, f, indent=2)

    print("Saved the best model to tourism_project/deployment/")
