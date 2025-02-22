import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress convergence warnings for clarity in this baseline script
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# 1. Load the Training Data
# -----------------------------
train = pd.read_csv(r"C:\Users\arche\code\CCAC_Competition_25\bracket_training (1).csv")

# -----------------------------
# 2. Preprocessing Training Data
# -----------------------------
# Select features to use for prediction.
# In this baseline, we use geographic info and regional winner predictions.
feature_cols = [
    'CustomerPostalCodeLatitude', 
    'CustomerPostalCodeLongitude', 
    'CustomerDMACode',
    'RegionWinner_East', 
    'RegionWinner_West', 
    'RegionWinner_South', 
    'RegionWinner_Midwest'
]

# Fill missing values in the selected feature columns with the median
train[feature_cols] = train[feature_cols].fillna(train[feature_cols].median())

# Define target columns (these are what we need to predict)
target_cols = [
    'SemifinalWinner_East_West', 
    'SemifinalWinner_South_Midwest', 
    'NationalChampion'
]

# (Optional) Handle any missing target values if necessary
# Here we simply drop rows with missing targets.
train.dropna(subset=target_cols, inplace=True)

# Standardize the features so they are on a similar scale.
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])

# -----------------------------
# 3. Fit Logistic Regression Models
# -----------------------------
# We will train one separate logistic regression model for each target.
models = {}
for target in target_cols:
    # Use multinomial logistic regression (suitable for multi-class targets)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, train[target])
    models[target] = model
    print(f"Trained logistic regression model for target: {target}")

# -----------------------------
# 4. Load and Preprocess Test Data
# -----------------------------
test = pd.read_csv(r"C:\Users\arche\code\CCAC_Competition_25\bracket_test.csv")

# Fill missing values in test features with the median of each column
test[feature_cols] = test[feature_cols].fillna(test[feature_cols].median())

# Apply the same scaling as the training data
X_test = scaler.transform(test[feature_cols])

# -----------------------------
# 5. Generate Predictions for Test Data
# -----------------------------
# For each target, predict the class (i.e. team id) using the corresponding model.
predictions = {}
for target in target_cols:
    predictions[target] = models[target].predict(X_test)
    print(f"Generated predictions for target: {target}")

# -----------------------------
# 6. Create Submission File
# -----------------------------
# The submission template requires:
# BracketEntryId, SemifinalWinner_South_Midwest, SemifinalWinner_East_West, NationalChampion
submission = pd.DataFrame({
    "BracketEntryId": test["BracketEntryId"],
    "SemifinalWinner_South_Midwest": predictions["SemifinalWinner_South_Midwest"],
    "SemifinalWinner_East_West": predictions["SemifinalWinner_East_West"],
    "NationalChampion": predictions["NationalChampion"]
})

# Save the submission to CSV
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
