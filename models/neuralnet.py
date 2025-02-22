import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Check for GPU availability
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 1. Load and Preprocess Training Data
# -----------------------------
# Load training CSV
train = pd.read_csv(r"C:\Users\arche\code\CCAC_Competition_25\bracket_training (1).csv")

# Define feature columns and fill missing values with median
feature_cols = [
    'CustomerPostalCodeLatitude', 
    'CustomerPostalCodeLongitude', 
    'CustomerDMACode',
    'RegionWinner_East', 
    'RegionWinner_West', 
    'RegionWinner_South', 
    'RegionWinner_Midwest'
]
train[feature_cols] = train[feature_cols].fillna(train[feature_cols].median())

# Define target columns and drop rows with missing targets
target_cols = ['SemifinalWinner_East_West', 'SemifinalWinner_South_Midwest', 'NationalChampion']
train.dropna(subset=target_cols, inplace=True)

# Label encode each target; store label encoders and number of classes
label_encoders = {}
num_classes = {}
for target in target_cols:
    le = LabelEncoder()
    train[target] = le.fit_transform(train[target])
    label_encoders[target] = le
    num_classes[target] = len(le.classes_)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(train[feature_cols])
y1 = train[target_cols[0]].values  # SemifinalWinner_East_West
y2 = train[target_cols[1]].values  # SemifinalWinner_South_Midwest
y3 = train[target_cols[2]].values  # NationalChampion

# Create train/validation split
X_train, X_val, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val = train_test_split(
    X, y1, y2, y3, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y1_train_tensor = torch.tensor(y1_train, dtype=torch.long).to(device)
y2_train_tensor = torch.tensor(y2_train, dtype=torch.long).to(device)
y3_train_tensor = torch.tensor(y3_train, dtype=torch.long).to(device)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y1_val_tensor = torch.tensor(y1_val, dtype=torch.long).to(device)
y2_val_tensor = torch.tensor(y2_val, dtype=torch.long).to(device)
y3_val_tensor = torch.tensor(y3_val, dtype=torch.long).to(device)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y1_train_tensor, y2_train_tensor, y3_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y1_val_tensor, y2_val_tensor, y3_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 2. Define the Multi-Output Model
# -----------------------------
class MultiOutputNet(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2, num_classes3):
        super(MultiOutputNet, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Separate output heads for each target
        self.head1 = nn.Linear(64, num_classes1)  # SemifinalWinner_East_West
        self.head2 = nn.Linear(64, num_classes2)  # SemifinalWinner_South_Midwest
        self.head3 = nn.Linear(64, num_classes3)  # NationalChampion
    
    def forward(self, x):
        x = self.shared(x)
        out1 = self.head1(x)
        out2 = self.head2(x)
        out3 = self.head3(x)
        return out1, out2, out3

input_dim = X_train.shape[1]
model = MultiOutputNet(input_dim,
                       num_classes[target_cols[0]],
                       num_classes[target_cols[1]],
                       num_classes[target_cols[2]])
model.to(device)  # Move model to GPU if available

# -----------------------------
# 3. Define Loss and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 4. Training Loop
# -----------------------------
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, t1, t2, t3 in train_loader:
        # Move batch to device
        inputs, t1, t2, t3 = inputs.to(device), t1.to(device), t2.to(device), t3.to(device)
        
        optimizer.zero_grad()
        out1, out2, out3 = model(inputs)
        loss1 = criterion(out1, t1)
        loss2 = criterion(out2, t2)
        loss3 = criterion(out3, t3)
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct1 = correct2 = correct3 = total = 0
    with torch.no_grad():
        for inputs, t1, t2, t3 in val_loader:
            inputs, t1, t2, t3 = inputs.to(device), t1.to(device), t2.to(device), t3.to(device)
            out1, out2, out3 = model(inputs)
            loss1 = criterion(out1, t1)
            loss2 = criterion(out2, t2)
            loss3 = criterion(out3, t3)
            loss = loss1 + loss2 + loss3
            val_loss += loss.item() * inputs.size(0)
            _, pred1 = torch.max(out1, 1)
            _, pred2 = torch.max(out2, 1)
            _, pred3 = torch.max(out3, 1)
            correct1 += (pred1 == t1).sum().item()
            correct2 += (pred2 == t2).sum().item()
            correct3 += (pred3 == t3).sum().item()
            total += inputs.size(0)
    val_loss /= len(val_dataset)
    acc1 = correct1 / total
    acc2 = correct2 / total
    acc3 = correct3 / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Acc1: {acc1:.4f}, Acc2: {acc2:.4f}, Acc3: {acc3:.4f}")

# -----------------------------
# 5. Load and Preprocess Test Data for Prediction
# -----------------------------
test = pd.read_csv(r"C:\Users\arche\code\CCAC_Competition_25\bracket_test.csv")
test[feature_cols] = test[feature_cols].fillna(test[feature_cols].median())
X_test = scaler.transform(test[feature_cols])
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# -----------------------------
# 6. Generate Predictions and Create Submission
# -----------------------------
model.eval()
with torch.no_grad():
    out1, out2, out3 = model(X_test_tensor)
    pred1 = torch.argmax(out1, dim=1).cpu().numpy()
    pred2 = torch.argmax(out2, dim=1).cpu().numpy()
    pred3 = torch.argmax(out3, dim=1).cpu().numpy()

# Convert encoded predictions back to original labels
pred1_labels = label_encoders[target_cols[0]].inverse_transform(pred1)
pred2_labels = label_encoders[target_cols[1]].inverse_transform(pred2)
pred3_labels = label_encoders[target_cols[2]].inverse_transform(pred3)

submission = pd.DataFrame({
    "BracketEntryId": test["BracketEntryId"],
    "SemifinalWinner_South_Midwest": pred2_labels,
    "SemifinalWinner_East_West": pred1_labels,
    "NationalChampion": pred3_labels
})
submission.to_csv("submission_pytorch_gpu.csv", index=False)
print("Submission saved to submission_pytorch_gpu.csv")
