import numpy as np
import pandas as pd
import ast
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Add parent directory to path so we can import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.load_ecg_signals import scp_to_multiclass

# Paths
EMBEDDING_PATH = "../data/processed/roberta_embeddings.npy"
CSV_PATH = "../data/ptbxl/ptbxl_database.csv"
MODEL_SAVE_PATH = "../results/models/drf_roberta_model.pkl"

# Config
TARGET_CLASSES = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']

print("🧠 Training DRF on RoBERTa Embeddings for Cardiovascular Classification")
print("=" * 65)

# Step 1: Load embeddings and metadata
print("📥 Loading RoBERTa embeddings...")
X = np.load(EMBEDDING_PATH)
print(f"   Embeddings shape: {X.shape}")

print("📋 Loading metadata...")
df = pd.read_csv(CSV_PATH)
df = df[df['scp_codes'].notna()]
df = df[df['scp_codes'].str.contains('|'.join(TARGET_CLASSES))]
df = df.drop_duplicates(subset="ecg_id").reset_index(drop=True)
print(f"   Metadata records: {len(df)}")

# Step 2: Get labels
print("🏷️  Converting diagnostic codes to multiclass labels...")
y_raw = df['scp_codes'].values
y = scp_to_multiclass(y_raw, target_classes=TARGET_CLASSES)

# Sanity check
if len(y) != X.shape[0]:
    raise ValueError(f"⚠️ Label/embedding mismatch: {len(y)} labels vs {X.shape[0]} embeddings")

# Filter out -1 (unmatched codes)
mask = y != -1
X = X[mask]
y = y[mask]

print(f"✅ Final dataset: {X.shape[0]} samples, {len(np.unique(y))} classes")

# Show class distribution
unique, counts = np.unique(y, return_counts=True)
print("\n📊 Class distribution:")
for cls, count in zip(unique, counts):
    print(f"   {TARGET_CLASSES[cls]} ({cls}): {count} samples")

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(f"\n🔀 Split: {len(X_train)} train / {len(X_test)} test")

# Step 4: Train DRF classifier
print("\n🌲 Training Dynamic Random Forest...")
clf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=20, 
    random_state=42, 
    class_weight='balanced',
    n_jobs=-1  # Use all CPU cores
)
clf.fit(X_train, y_train)
print(f"   Model trained with {clf.n_estimators} trees")

# Step 5: Evaluate
print("\n📊 Evaluating model...")
y_pred = clf.predict(X_test)

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate accuracy
accuracy = (y_pred == y_test).mean()
print(f"\n🎯 Overall Accuracy: {accuracy:.1%}")

# Step 6: Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(clf, MODEL_SAVE_PATH)
print(f"💾 Saved model to {MODEL_SAVE_PATH}")

print("\n✅ DRF Training Complete!")
print("🔬 Next: Implement WPSA optimization for hybrid WPSA-DRF model")
