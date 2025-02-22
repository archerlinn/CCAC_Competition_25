import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# Suppress warnings for clarity
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# 1. Load the Training Data
# -----------------------------
train = pd.read_csv(r"C:\Users\arche\code\CCAC_Competition_25\bracket_training (1).csv")

# -----------------------------
# 2. Preprocessing Training Data
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
# Fill missing values with the median
train[feature_cols] = train[feature_cols].fillna(train[feature_cols].median())

# Define target columns
target_cols = [
    'SemifinalWinner_East_West', 
    'SemifinalWinner_South_Midwest', 
    'NationalChampion'
]
# Drop rows with missing targets
train.dropna(subset=target_cols, inplace=True)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])

# -----------------------------
# 3. Train XGBoost Models with Label Encoding
# -----------------------------
# We'll store both the model and its corresponding label encoder for each target.
models = {}
label_encoders = {}

for target in target_cols:
    y_train = train[target]
    
    # Encode target labels to 0-indexed integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    model = xgb.XGBClassifier(
        n_estimators=500,      # More trees can help performance
        learning_rate=0.05,    # Lower learning rate for better generalization
        max_depth=6,           # Depth to capture interactions
        subsample=0.8,         # Subsample ratio to prevent overfitting
        colsample_bytree=0.8,  
        objective='multi:softmax', # Multi-class classification
        num_class=len(le.classes_),# Number of classes for this target
        random_state=42
    )
    model.fit(X_train, y_train_enc)
    
    models[target] = model
    label_encoders[target] = le
    print(f"Trained XGBoost model for target: {target}")

# -----------------------------
# 4. Load and Preprocess Test Data
# -----------------------------
test = pd.read_csv(r"C:\Users\arche\code\CCAC_Competition_25\bracket_test.csv")
test[feature_cols] = test[feature_cols].fillna(test[feature_cols].median())
X_test = scaler.transform(test[feature_cols])

# -----------------------------
# 5. Generate Predictions for Test Data
# -----------------------------
predictions = {}
for target in target_cols:
    model = models[target]
    le = label_encoders[target]
    # Predict encoded labels and transform them back to the original labels
    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc.astype(int))
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
