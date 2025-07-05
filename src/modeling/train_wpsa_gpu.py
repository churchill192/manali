import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import joblib
import wfdb
from tqdm import tqdm
import xgboost as xgb
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
MODEL_SAVE_PATH = "../results/models/wpsa_gpu_optimized_model.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 GPU-Accelerated WPSA with XGBoost")
print("=" * 50)

# Check GPU availability
print(f"🔍 Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not available, using CPU")

# Load data (same as fusion model)
print("\n📥 Loading fusion features...")
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

# Load ECG features using pre-trained CNN
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

print("🔬 Extracting ECG features...")
cnn_model = ECGFeatureExtractor().to(DEVICE)
cnn_model.load_state_dict(torch.load("../results/models/cnn_ecg_model.pth", map_location=DEVICE), strict=False)
cnn_model.eval()

# Extract ECG features
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

# 🚀 GPU-Accelerated WPSA Algorithm with XGBoost
class GPUWolfPackSearchAlgorithm:
    def __init__(self, pack_size=6, max_iterations=10, bounds=None):
        self.pack_size = pack_size
        self.max_iterations = max_iterations
        self.bounds = bounds or {
            'n_estimators': (100, 800),
            'max_depth': (3, 15),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (0, 10),
            'reg_lambda': (0, 10)
        }
        
        # Wolf hierarchy
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.pack = []
        
        print(f"🐺 GPU-WPSA initialized: Pack size={pack_size}, Iterations={max_iterations}")

    def initialize_pack(self):
        """Initialize wolf pack with random positions"""
        self.pack = []
        for _ in range(self.pack_size):
            wolf = {}
            for param, (min_val, max_val) in self.bounds.items():
                if param == 'n_estimators' or param == 'max_depth':
                    wolf[param] = random.randint(int(min_val), int(max_val))
                else:
                    wolf[param] = random.uniform(min_val, max_val)
            self.pack.append(wolf)

    def evaluate_fitness(self, wolf_params, X, y):
        """Evaluate wolf fitness using GPU-accelerated XGBoost"""
        try:
            # Use GPU if available
            tree_method = 'gpu_hist' if torch.cuda.is_available() else 'hist'
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=wolf_params['n_estimators'],
                max_depth=wolf_params['max_depth'],
                learning_rate=wolf_params['learning_rate'],
                subsample=wolf_params['subsample'],
                colsample_bytree=wolf_params['colsample_bytree'],
                reg_alpha=wolf_params['reg_alpha'],
                reg_lambda=wolf_params['reg_lambda'],
                tree_method=tree_method,
                gpu_id=0 if torch.cuda.is_available() else None,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            
            # Use 3-fold CV for speed
            scores = cross_val_score(xgb_model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        except Exception as e:
            print(f"⚠️  Wolf evaluation failed: {e}")
            return 0.0

    def update_pack_hierarchy(self, fitnesses):
        """Update alpha, beta, gamma wolves based on fitness"""
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        self.alpha = deepcopy(self.pack[sorted_indices[0]])
        self.beta = deepcopy(self.pack[sorted_indices[1]])
        self.gamma = deepcopy(self.pack[sorted_indices[2]])

    def update_wolf_position(self, wolf, alpha, beta, gamma):
        """Update wolf position based on alpha, beta, gamma"""
        updated_wolf = {}
        
        for param in wolf.keys():
            # WPSA position update equations
            A1 = 2 * random.random() - 1
            A2 = 2 * random.random() - 1
            A3 = 2 * random.random() - 1
            
            C1 = 2 * random.random()
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
            if param in ['n_estimators', 'max_depth']:
                new_position = int(new_position)
            
            updated_wolf[param] = new_position
        
        return updated_wolf

    def optimize(self, X, y):
        """Main GPU-WPSA optimization loop"""
        print("🚀 Starting GPU-accelerated WPSA optimization...")
        
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
                
                print(f"   Wolf {i+1}: {fitness:.4f}")
            
            # Update pack hierarchy
            self.update_pack_hierarchy(fitnesses)
            
            # Update wolf positions
            new_pack = []
            for wolf in self.pack:
                new_wolf = self.update_wolf_position(wolf, self.alpha, self.beta, self.gamma)
                new_pack.append(new_wolf)
            
            self.pack = new_pack
            
            print(f"   🏆 Best fitness: {best_fitness:.4f}")
            print(f"   🐺 Alpha: {self.alpha}")
        
        return best_params, best_fitness

# Run GPU-WPSA Optimization
print("\n🐺 Starting GPU-accelerated Wolf Pack Search...")
gpu_wpsa = GPUWolfPackSearchAlgorithm(pack_size=6, max_iterations=8)  # Faster settings
best_params, best_fitness = gpu_wpsa.optimize(X_train, y_train)

print(f"\n🏆 GPU-WPSA Optimization Complete!")
print(f"   Best CV accuracy: {best_fitness:.4f}")
print(f"   Best parameters: {best_params}")

# Train final model with optimized parameters
print("\n🌲 Training final XGBoost model...")
tree_method = 'gpu_hist' if torch.cuda.is_available() else 'hist'

final_model = xgb.XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    tree_method=tree_method,
    gpu_id=0 if torch.cuda.is_available() else None,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

final_model.fit(X_train, y_train)

# Final evaluation
y_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Final Test Results:")
print(f"🎯 GPU-WPSA Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

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
print(f"   - GPU-WPSA: {final_accuracy*100:.2f}%")
print(f"   - Improvement: {(final_accuracy - 0.8075)*100:.2f}%")
print(f"   - Gap to 92.5%: {(0.925 - final_accuracy)*100:.2f}%")

print("\n✅ GPU-WPSA Complete - Much faster than CPU!") 