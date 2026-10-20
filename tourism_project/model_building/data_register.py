# Data registration: check that the tourism dataset is in the repo and looks
# the way we expect, then print a quick summary of it.
import sys
import pandas as pd

DATASET_PATH = "tourism_project/data/tourism.csv"

# The columns we expect, taken from the project's data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

# Read the dataset straight from the repo's data folder
df = pd.read_csv(DATASET_PATH)
print(f"Loaded {DATASET_PATH}")

# Is any expected column missing? If so, stop here so the pipeline
# doesn't go on to train a model on broken data.
missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
if missing_cols:
    print(f"Validation failed - missing columns: {missing_cols}")
    sys.exit(1)

# The target should only ever be 0 (not purchased) or 1 (purchased)
if not df["ProdTaken"].isin([0, 1]).all():
    print("Validation failed - ProdTaken must contain only 0 and 1")
    sys.exit(1)

# Everything looks good, so print a short summary of the data
print("Validation passed - all expected columns are present")
print(f"Rows and columns : {df.shape}")
print(f"Missing values   : {df.isna().sum().sum()}")
print(f"Duplicate rows   : {df.duplicated().sum()}")
print(f"Purchase rate    : {df['ProdTaken'].mean():.2%}")
print("\nCustomers by purchase (0 = No, 1 = Yes):")
print(df["ProdTaken"].value_counts().to_string())
print("\nDataset registered successfully.")
