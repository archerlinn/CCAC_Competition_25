import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for clarity
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# 1. Load the Training Data
# -----------------------------
train = pd.read_csv(r"C:\Users\arche\code\CCAC_Competition_25\bracket_training (1).csv")

# -----------------------------
# 2. Preprocess Training Data
# -----------------------------
# Define features to use for prediction.
feature_cols = [
    'CustomerPostalCodeLatitude', 
    'CustomerPostalCodeLongitude', 
    'CustomerDMACode',
    'RegionWinner_East', 
    'RegionWinner_West', 
    'RegionWinner_South', 
    'RegionWinner_Midwest'
]
# Fill missing values in the selected features with the median.
train[feature_cols] = train[feature_cols].fillna(train[feature_cols].median())

# Define target columns.
target_cols = ['SemifinalWinner_East_West', 'SemifinalWinner_South_Midwest', 'NationalChampion']

# Drop rows with missing targets.
train.dropna(subset=target_cols, inplace=True)

# Use the raw features (CatBoost typically does not require scaling)
X_train = train[feature_cols].copy()

# -----------------------------
# 3. Train CatBoost Models with Label Encoding
# -----------------------------
# We will train a separate CatBoost model for each target.
models = {}
label_encoders = {}

for target in target_cols:
    y_train = train[target]
    
    # Encode target labels to 0-indexed integers.
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    label_encoders[target] = le
    
    # Define the CatBoost model.
    model = CatBoostClassifier(
        iterations=500,      # Number of boosting iterations.
        learning_rate=0.03,    # Lower learning rate for stability.
        depth=6,               # Tree depth to capture interactions.
        loss_function='MultiClass',  # For multi-class classification.
        verbose=0,             # Turn off verbose output.
        random_seed=42
    )
    
    # Train the model.
    model.fit(X_train, y_train_enc)
    models[target] = model
    print(f"Trained CatBoost model for target: {target}")

# -----------------------------
# 4. Load and Preprocess Test Data
# -----------------------------
test = pd.read_csv(r"C:\Users\arche\code\CCAC_Competition_25\bracket_test.csv")
test[feature_cols] = test[feature_cols].fillna(test[feature_cols].median())
X_test = test[feature_cols].copy()

# -----------------------------
# 5. Generate Predictions for Test Data
# -----------------------------
predictions = {}
for target in target_cols:
    model = models[target]
    le = label_encoders[target]
    
    # Predict the encoded labels.
    y_pred_enc = model.predict(X_test)
    # Convert predictions back to the original labels.
    y_pred = le.inverse_transform(y_pred_enc.astype(int).flatten())
    predictions[target] = y_pred
    print(f"Generated predictions for target: {target}")

# -----------------------------
# 6. Create Submission File
# -----------------------------
submission = pd.DataFrame({
    "BracketEntryId": test["BracketEntryId"],
    "SemifinalWinner_South_Midwest": predictions["SemifinalWinner_South_Midwest"],
    "SemifinalWinner_East_West": predictions["SemifinalWinner_East_West"],
    "NationalChampion": predictions["NationalChampion"]
})

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
