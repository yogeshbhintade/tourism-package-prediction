# Data preparation: clean the dataset and split it into training and test sets
# that the model-training step can use.
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the dataset from the repo's data folder
df = pd.read_csv("tourism_project/data/tourism.csv")
print(f"Loaded dataset: {df.shape}")

# These columns don't describe a customer before the first sales contact.
# Keeping them would let the model use information from the interaction.
post_contact_columns = [
    "TypeofContact",
    "ProductPitched",
    "DurationOfPitch",
    "NumberOfFollowups",
    "PitchSatisfactionScore",
]

# Remove export leftovers, customer IDs, and fields unavailable at prediction time.
columns_to_drop = ["Unnamed: 0", "CustomerID"] + post_contact_columns
columns_to_drop = [column for column in columns_to_drop if column in df.columns]
df = df.drop(columns=columns_to_drop)

# Fix spelling issues 'Fe Male' and 'Free Lancer' so each category is written one way only
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})

# With the IDs gone, some rows are exact copies of others; keep one of each
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows, {len(df)} rows left")

# Separate the customer details (X) from what we want to predict (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Keep 80% of the rows for training and 20% for testing. stratify=y keeps the
# share of buyers (about 19%) the same in both sets.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the four files; the GitHub workflow hands them to the training job
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Training set: {Xtrain.shape}, buyers {ytrain.mean():.1%}")
print(f"Test set    : {Xtest.shape}, buyers {ytest.mean():.1%}")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv and ytest.csv")
