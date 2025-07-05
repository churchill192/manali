import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import joblib
import wfdb
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import random
from copy import deepcopy

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.load_ecg_signals import scp_to_multiclass

# Configuration
TARGET_CLASSES = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']
ROBERTA_PATH = "../data/processed/roberta_embeddings.npy"
CSV_PATH = "../data/ptbxl/ptbxl_database.csv"
ECG_BASE_PATH = "../data/ptbxl/"
MODEL_SAVE_PATH = "../results/models/wpsa_optimized_model.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🐺 Wolf Pack Search Algorithm (WPSA) - Real Implementation")
print("=" * 60)

# Load data (same as fusion model)
print("📥 Loading fusion features...")
roberta_embeddings = np.load(ROBERTA_PATH)
df = pd.read_csv(CSV_PATH)
df = df[df['scp_codes'].notna()]
df = df[df['scp_codes'].str.contains('|'.join(TARGET_CLASSES))]
df = df.drop_duplicates(subset="ecg_id").reset_index(drop=True)

y_raw = df['scp_codes'].values
y = scp_to_multiclass(y_raw, target_classes=TARGET_CLASSES)

mask = y != -1
df_filtered = df[mask].reset_index(drop=True)
y_filtered = y[mask]
roberta_filtered = roberta_embeddings[mask]

print(f"✅ Dataset size: {len(df_filtered)} records")

# Load pre-trained ECG features (from fusion model)
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
        self.fc = nn.Linear(64 * 250, 128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        return self.fc(x)

print("🔬 Extracting ECG features (this may take a minute)...")
cnn_model = ECGFeatureExtractor().to(DEVICE)
cnn_model.load_state_dict(torch.load("../results/models/cnn_ecg_model.pth", map_location=DEVICE), strict=False)
cnn_model.eval()

# Quick ECG feature extraction (show progress every 1000 records)
ecg_features = []
for idx, row in enumerate(tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing ECG")):
    _, row_data = row
    try:
        filename_lr = str(row_data['filename_lr'])
        full_path = os.path.join(ECG_BASE_PATH, filename_lr)
        record = wfdb.rdrecord(full_path)
        signal = record.p_signal.T
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = cnn_model(signal_tensor).cpu().numpy().flatten()
            ecg_features.append(features)
    except:
        ecg_features.append(np.zeros(128))

ecg_features = np.array(ecg_features)
X_fusion = np.concatenate([ecg_features, roberta_filtered], axis=1)

print(f"✅ Fusion features ready: {X_fusion.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_fusion, y_filtered, stratify=y_filtered, test_size=0.2, random_state=42
)

print(f"🔀 Split: {len(X_train)} train / {len(X_test)} test")

# 🐺 WPSA Algorithm Implementation
class WolfPackSearchAlgorithm:
    def __init__(self, pack_size=10, max_iterations=20, bounds=None):
        self.pack_size = pack_size
        self.max_iterations = max_iterations
        self.bounds = bounds or {
            'n_estimators': (50, 500),
            'max_depth': (10, 50),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': (0.3, 1.0)
        }
        
        # Wolf roles
        self.alpha = None  # Best wolf (best solution)
        self.beta = None   # Second best
        self.gamma = None  # Third best
        self.pack = []     # All wolves
        
        print(f"🐺 WPSA initialized: Pack size={pack_size}, Iterations={max_iterations}")

    def initialize_pack(self):
        """Initialize wolf pack with random positions"""
        self.pack = []
        for _ in range(self.pack_size):
            wolf = {}
            for param, (min_val, max_val) in self.bounds.items():
                if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    wolf[param] = random.randint(int(min_val), int(max_val))
                else:
                    wolf[param] = random.uniform(min_val, max_val)
            self.pack.append(wolf)

    def evaluate_fitness(self, wolf_params, X, y):
        """Evaluate wolf fitness using cross-validation"""
        try:
            rf = RandomForestClassifier(
                n_estimators=wolf_params['n_estimators'],
                max_depth=wolf_params['max_depth'],
                min_samples_split=wolf_params['min_samples_split'],
                min_samples_leaf=wolf_params['min_samples_leaf'],
                max_features=wolf_params['max_features'],
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            # Use 3-fold CV for speed
            scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        except:
            return 0.0  # Return poor fitness if model fails

    def update_pack_hierarchy(self, fitnesses):
        """Update alpha, beta, gamma wolves based on fitness"""
        # Sort wolves by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        self.alpha = deepcopy(self.pack[sorted_indices[0]])
        self.beta = deepcopy(self.pack[sorted_indices[1]])
        self.gamma = deepcopy(self.pack[sorted_indices[2]])

    def update_wolf_position(self, wolf, alpha, beta, gamma):
        """Update wolf position based on alpha, beta, gamma"""
        updated_wolf = {}
        
        for param in wolf.keys():
            # Calculate distances to alpha, beta, gamma
            A1 = 2 * random.random() - 1  # Random coefficient
            A2 = 2 * random.random() - 1
            A3 = 2 * random.random() - 1
            
            C1 = 2 * random.random()  # Random coefficient
            C2 = 2 * random.random()
            C3 = 2 * random.random()
            
            # Distance calculations
            D_alpha = abs(C1 * alpha[param] - wolf[param])
            D_beta = abs(C2 * beta[param] - wolf[param])
            D_gamma = abs(C3 * gamma[param] - wolf[param])
            
            # Position updates
            X1 = alpha[param] - A1 * D_alpha
            X2 = beta[param] - A2 * D_beta
            X3 = gamma[param] - A3 * D_gamma
            
            # Average position
            new_position = (X1 + X2 + X3) / 3
            
            # Apply bounds
            min_val, max_val = self.bounds[param]
            new_position = max(min_val, min(max_val, new_position))
            
            # Convert to integer for discrete parameters
            if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                new_position = int(new_position)
            
            updated_wolf[param] = new_position
        
        return updated_wolf

    def optimize(self, X, y):
        """Main WPSA optimization loop"""
        print("🚀 Starting WPSA optimization...")
        
        # Initialize pack
        self.initialize_pack()
        
        best_fitness = 0.0
        best_params = None
        
        for iteration in range(self.max_iterations):
            print(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Evaluate all wolves
            fitnesses = []
            for i, wolf in enumerate(self.pack):
                fitness = self.evaluate_fitness(wolf, X, y)
                fitnesses.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = deepcopy(wolf)
            
            # Update pack hierarchy
            self.update_pack_hierarchy(fitnesses)
            
            # Update wolf positions
            new_pack = []
            for wolf in self.pack:
                new_wolf = self.update_wolf_position(wolf, self.alpha, self.beta, self.gamma)
                new_pack.append(new_wolf)
            
            self.pack = new_pack
            
            print(f"   Best fitness: {best_fitness:.4f}")
            print(f"   Alpha params: {self.alpha}")
        
        return best_params, best_fitness

# Run WPSA Optimization
print("\n🐺 Starting Wolf Pack Search Algorithm...")
wpsa = WolfPackSearchAlgorithm(pack_size=8, max_iterations=15)  # Reduced for faster execution
best_params, best_fitness = wpsa.optimize(X_train, y_train)

print(f"\n🏆 WPSA Optimization Complete!")
print(f"   Best CV accuracy: {best_fitness:.4f}")
print(f"   Best parameters: {best_params}")

# Train final model with optimized parameters
print("\n🌲 Training final model with WPSA-optimized parameters...")
final_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

# Final evaluation
y_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Final Test Results:")
print(f"🎯 WPSA-Optimized Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(final_model, MODEL_SAVE_PATH)

print(f"\n💾 Model saved to: {MODEL_SAVE_PATH}")

print("\n🎯 Performance Comparison:")
print("   - Fusion (baseline): 80.75%")
print(f"   - WPSA-Optimized: {final_accuracy*100:.2f}%")
print(f"   - Improvement: {(final_accuracy - 0.8075)*100:.2f}%")
print(f"   - Gap to 92.5%: {(0.925 - final_accuracy)*100:.2f}%")

print("\n✅ WPSA Implementation Complete!") 