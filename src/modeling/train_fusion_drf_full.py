import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import joblib
import wfdb
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.load_ecg_signals import scp_to_multiclass

# Configuration
TARGET_CLASSES = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']
ROBERTA_PATH = "../data/processed/roberta_embeddings.npy"
CSV_PATH = "../data/ptbxl/ptbxl_database.csv"
ECG_BASE_PATH = "../data/ptbxl/"
FUSION_MODEL_SAVE_PATH = "../results/models/fusion_drf_full_model.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔗 Multi-Modal Fusion: Full Dataset (17,162 records)")
print("=" * 60)

# Step 1: Load RoBERTa embeddings (17,200 total)
print("📥 Loading RoBERTa embeddings...")
roberta_embeddings = np.load(ROBERTA_PATH)
print(f"   RoBERTa embeddings shape: {roberta_embeddings.shape}")

# Step 2: Load metadata using SAME filtering as RoBERTa model
print("📋 Loading and filtering metadata...")
df = pd.read_csv(CSV_PATH)
df = df[df['scp_codes'].notna()]
df = df[df['scp_codes'].str.contains('|'.join(TARGET_CLASSES))]
df = df.drop_duplicates(subset="ecg_id").reset_index(drop=True)
print(f"   Filtered metadata: {len(df)} records")

# Step 3: Generate labels (same as RoBERTa model)
print("🏷️  Converting diagnostic codes to labels...")
y_raw = df['scp_codes'].values
y = scp_to_multiclass(y_raw, target_classes=TARGET_CLASSES)

# Filter out -1 (unmatched codes) - same as RoBERTa model
mask = y != -1
df_filtered = df[mask].reset_index(drop=True)
y_filtered = y[mask]
roberta_filtered = roberta_embeddings[mask]

print(f"✅ Final dataset size: {len(df_filtered)} records")
print(f"   RoBERTa embeddings: {roberta_filtered.shape}")
print(f"   Labels: {y_filtered.shape}")

# Step 4: Define CNN Feature Extractor (same architecture as train_cnn.py)
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
        self.fc = nn.Linear(64 * 250, 128)  # 128-dim ECG features

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        return self.fc(x)

# Step 5: Load pre-trained CNN model
print("🧠 Loading pre-trained CNN model...")
cnn_model = ECGFeatureExtractor().to(DEVICE)
cnn_model.load_state_dict(torch.load("../results/models/cnn_ecg_model.pth", map_location=DEVICE), strict=False)
cnn_model.eval()
print("✅ CNN model loaded successfully")

# Step 6: Extract ECG features for all 17,162 records
print("🔬 Extracting ECG features for all records...")
ecg_features = []
failed_records = 0

for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing ECG signals"):
    try:
        # Build file path
        filename_lr = str(row['filename_lr'])
        full_path = os.path.join(ECG_BASE_PATH, filename_lr)
        
        # Load ECG signal
        record = wfdb.rdrecord(full_path)
        signal = record.p_signal.T  # Shape: (12, 1000)
        
        # Convert to tensor and extract features
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = cnn_model(signal_tensor).cpu().numpy().flatten()
            ecg_features.append(features)
            
    except Exception as e:
        # Handle missing files by using zero features
        ecg_features.append(np.zeros(128))
        failed_records += 1

ecg_features = np.array(ecg_features)
print(f"✅ ECG features extracted: {ecg_features.shape}")
print(f"   Failed records: {failed_records}")

# Step 7: Combine ECG + RoBERTa features
print("🔗 Combining ECG and RoBERTa features...")
X_fusion = np.concatenate([ecg_features, roberta_filtered], axis=1)
print(f"✅ Fusion features shape: {X_fusion.shape}")
print(f"   ECG features: {ecg_features.shape[1]} dimensions")
print(f"   RoBERTa features: {roberta_filtered.shape[1]} dimensions")

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_fusion, y_filtered, stratify=y_filtered, test_size=0.2, random_state=42
)
print(f"🔀 Dataset split: {len(X_train)} train / {len(X_test)} test")

# Step 9: Train Fusion Random Forest
print("🌲 Training Fusion Random Forest...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print("✅ Training completed!")

# Step 10: Evaluation
print("\n📊 Evaluating fusion model...")
y_pred = clf.predict(X_test)

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate accuracy
accuracy = (y_pred == y_test).mean()
print(f"\n🎯 Fusion Model Accuracy: {accuracy:.2%}")

# Show class distribution
unique, counts = np.unique(y_test, return_counts=True)
print(f"\n📊 Test set class distribution:")
for cls, count in zip(unique, counts):
    print(f"   {TARGET_CLASSES[cls]}: {count} samples")

# Step 11: Save model
os.makedirs(os.path.dirname(FUSION_MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(clf, FUSION_MODEL_SAVE_PATH)
print(f"\n💾 Fusion model saved to: {FUSION_MODEL_SAVE_PATH}")

print("\n✅ Multi-Modal Fusion Training Complete!")
print("🎯 Comparison with previous models:")
print("   - CNN (ECG only): 42%")
print("   - DRF (RoBERTa only): 80%")
print(f"   - Fusion (ECG + RoBERTa): {accuracy:.2%}") 