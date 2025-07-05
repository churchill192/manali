import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.load_ecg_signals import (
    load_metadata,
    filter_df_by_target_classes,
    build_dataset,
    scp_to_multiclass
)

# Configuration
TARGET_CLASSES = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']
ROBERTA_PATH = "../data/processed/roberta_embeddings.npy"
FUSION_MODEL_SAVE_PATH = "../results/models/fusion_drf_model.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load RoBERTa embeddings
print("📥 Loading RoBERTa embeddings...")
roberta_embeddings = np.load(ROBERTA_PATH)

# 🧠 Define CNN model (same as train_cnn.py)
class ECGFeatureExtractor(nn.Module):
    def __init__(self):
        super(ECGFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 250, 128)  # output size: 128-dim ECG feature vector

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        return self.fc(x)

# Load metadata & ECG signals
df = load_metadata()
df_balanced = filter_df_by_target_classes(df, TARGET_CLASSES, samples_per_class=300)
X_ecg_raw, y_raw = build_dataset(df_balanced)
y_labels = scp_to_multiclass(y_raw, TARGET_CLASSES)

# Filter -1 labels
valid_mask = y_labels != -1
X_ecg_raw = X_ecg_raw[valid_mask]
y_labels = y_labels[valid_mask]
roberta_embeddings = roberta_embeddings[:len(y_labels)]  # align length

print(f"\n✅ ECG: {X_ecg_raw.shape}, RoBERTa: {roberta_embeddings.shape}, Labels: {y_labels.shape}")

# Step 1: Extract CNN features (disable gradient)
print("\n🔬 Extracting CNN features...")
model = ECGFeatureExtractor().to(DEVICE)
model.load_state_dict(torch.load("../results/models/cnn_ecg_model.pth", map_location=DEVICE), strict=False)
model.eval()

ecg_features = []
with torch.no_grad():
    for signal in torch.tensor(X_ecg_raw, dtype=torch.float32).to(DEVICE):
        signal = signal.unsqueeze(0)  # batch dim
        feat = model(signal).cpu().numpy().flatten()
        ecg_features.append(feat)
ecg_features = np.array(ecg_features)

print(f"✅ Extracted ECG features shape: {ecg_features.shape}")

# Step 2: Concatenate ECG + RoBERTa
print("🔗 Concatenating features...")
X_fusion = np.concatenate([ecg_features, roberta_embeddings[:len(ecg_features)]], axis=1)
print(f"✅ Final fusion shape: {X_fusion.shape}")

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_fusion, y_labels, stratify=y_labels, test_size=0.2, random_state=42
)

# Step 4: Train DRF
print("\n🌲 Training Fusion Random Forest...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = clf.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = (y_pred == y_test).mean()
print(f"\n🎯 Fusion Accuracy: {accuracy:.2%}")

# Step 6: Save
os.makedirs(os.path.dirname(FUSION_MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(clf, FUSION_MODEL_SAVE_PATH)
print(f"💾 Saved fusion model to: {FUSION_MODEL_SAVE_PATH}")
