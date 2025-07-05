import numpy as np
import pandas as pd
import ast
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.load_ecg_signals import scp_to_multiclass

# Constants
EMBEDDING_PATH = "../data/processed/roberta_embeddings.npy"
CSV_PATH = "../data/ptbxl/ptbxl_database.csv"
MODEL_SAVE_PATH = "../results/models/wpsa_drf_roberta_model.pkl"

TARGET_CLASSES = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']

print("🐺 Training WPSA-DRF Model (RoBERTa + Random Forest Optimized)")
print("=" * 65)

# Load embeddings
X = np.load(EMBEDDING_PATH)
print(f"📦 Loaded RoBERTa embeddings: {X.shape}")

# Load and filter metadata
df = pd.read_csv(CSV_PATH)
df = df[df['scp_codes'].notna()]
df = df[df['scp_codes'].str.contains('|'.join(TARGET_CLASSES))]
df = df.drop_duplicates(subset="ecg_id").reset_index(drop=True)
print(f"📋 Loaded metadata: {df.shape}")

# Generate labels
y_raw = df['scp_codes'].values
y = scp_to_multiclass(y_raw, target_classes=TARGET_CLASSES)

# Filter out unmatched labels
mask = y != -1
X = X[mask]
y = y[mask]

print(f"✅ Final dataset: {X.shape[0]} samples across {len(np.unique(y))} classes")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(f"🔀 Split: {len(X_train)} train / {len(X_test)} test")

# 🧠 Use WPSA-optimized hyperparameters
clf = RandomForestClassifier(
    n_estimators=108,        # WPSA-tuned
    max_depth=29,            # WPSA-tuned
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

print("\n🌲 Training Random Forest with WPSA-optimized hyperparameters...")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Accuracy
accuracy = (y_pred == y_test).mean()
print(f"\n🎯 Overall Accuracy: {accuracy:.2%}")

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(clf, MODEL_SAVE_PATH)
print(f"💾 Model saved to {MODEL_SAVE_PATH}")

print("\n✅ WPSA-DRF Training Complete!")
