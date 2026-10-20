# Wellness Tourism Package Prediction – MLOps Pipeline

"Visit with Us" wants to know, **before contacting a customer**, whether they are likely to buy the new
Wellness Tourism Package. This repository holds an end-to-end MLOps pipeline that registers the data,
prepares it, trains and tracks an XGBoost model, commits the best model back to the repo and serves it
through a Streamlit app.

## Repository structure

```
.github/workflows/pipeline.yml          # CI/CD: register -> prep -> train -> commit model
tourism_project/
├── data/tourism.csv                    # registered dataset
├── model_building/
│   ├── data_register.py                # validates expected columns, prints summary
│   ├── prep.py                         # cleaning + stratified 80/20 split
│   └── train.py                        # GridSearchCV XGBoost + MLflow tracking
├── deployment/
│   ├── app.py                          # Streamlit app
│   ├── requirements.txt                # app dependencies (Streamlit Community Cloud)
│   ├── best_tourism_model_v1.joblib    # committed by the pipeline
│   └── model_metadata.json             # threshold, best params, metrics
└── requirements.txt                    # pipeline dependencies (GitHub Actions)
```

## Pipeline

Every push to `main` runs three chained jobs in GitHub Actions:

1. **register-dataset** – validates `tourism.csv` and uploads it as the `registered-data` artifact.
2. **data-prep** – cleans the data, drops identifier columns, splits train/test and uploads the
   `data-splits` artifact.
3. **model-traning** – downloads the splits, tunes XGBoost with GridSearchCV while logging every
   parameter combination to MLflow, evaluates on the test set and commits the best model to
   `tourism_project/deployment/`.

The Streamlit app (`tourism_project/deployment/app.py`) is deployed on Streamlit Community Cloud and
loads the committed model directly from the repository.
