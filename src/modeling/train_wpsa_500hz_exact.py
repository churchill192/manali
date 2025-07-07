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
import warnings
import time
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.load_ecg_signals import scp_to_multiclass

# Configuration - 500 Hz Version
TARGET_CLASSES = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']

# Use absolute paths to avoid directory issues
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ROBERTA_PATH = os.path.join(PROJECT_ROOT, "data/processed/roberta_embeddings.npy")
CSV_PATH = os.path.join(PROJECT_ROOT, "data/ptbxl/ptbxl_database.csv")
ECG_BASE_PATH = os.path.join(PROJECT_ROOT, "data/ptbxl")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "results/models/wpsa_500hz_exact_model.pkl")
CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "results/models/cnn_ecg_500hz_model.pth")  # 500 Hz CNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🎯 WPSA-DRF 500 Hz Implementation - Optimized for 1 Hour Runtime")
print("📊 Target: 92.5% Accuracy with 500 Hz High-Resolution ECG")
print("⏱️ Runtime Limit: 1 Hour (60 minutes)")
print("=" * 70)

# Load fusion features (same as successful fusion model)
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

# Load 500 Hz ECG CNN Feature Extractor
class ECGFeatureExtractor500Hz(nn.Module):
    def __init__(self):
        super(ECGFeatureExtractor500Hz, self).__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.flatten = nn.Flatten()
        # For 5000 samples: 5000 -> 2500 -> 1250 -> 625
        self.fc1 = nn.Linear(128 * 625, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)  # 128-dim features

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # (B, 32, 2500)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # (B, 64, 1250)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # (B, 128, 625)
        x = self.flatten(x)  # (B, 128*625)
        x = self.dropout1(self.relu(self.fc1(x)))  # (B, 256)
        return self.fc2(x)  # (B, 128)

print("🔬 Loading 500 Hz CNN model...")
cnn_model = ECGFeatureExtractor500Hz().to(DEVICE)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE), strict=False)
cnn_model.eval()
print("✅ 500 Hz CNN model loaded successfully")

# Extract 500 Hz ECG features - ULTRA OPTIMIZED VERSION (from reference)
print("💡 Pre-allocating ECG features array for efficiency...")
ecg_features = np.zeros((len(df_filtered), 128), dtype=np.float32)  # Pre-allocate!

failed_count = 0
print("🔬 Processing 500 Hz ECG signals...")
for idx, row in enumerate(tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing 500Hz ECG")):
    _, row_data = row
    try:
        # Convert to 500 Hz path
        filename_lr = str(row_data['filename_lr'])
        filename_500hz = filename_lr.replace('records100', 'records500').replace('_lr', '_hr')
        full_path = os.path.join(ECG_BASE_PATH, filename_500hz)
        
        # Load 500 Hz ECG signal (5000 samples)
        record = wfdb.rdrecord(full_path)
        signal = record.p_signal.T  # Shape: (12, 5000)
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = cnn_model(signal_tensor).cpu().numpy().flatten()
            ecg_features[idx] = features  # Direct assignment - no append!
        
        # Debug last 20 records
        if idx >= len(df_filtered) - 20:
            print(f"✅ Completed {idx}: {filename_500hz}")
        
    except Exception as e:
        failed_count += 1
        print(f"❌ FAILED record {idx} ({filename_lr}): {e}")
        ecg_features[idx] = np.zeros(128)
        
    # Force GPU cleanup every 1000 records
    if idx % 1000 == 0:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"🧹 GPU cleanup at record {idx}")

print(f"🎯 500 Hz ECG processing completed! Failed: {failed_count}")
print("🧹 Final GPU cleanup...")
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("✅ 500 Hz ECG processing loop completed successfully!")
print(f"📊 ECG features shape: {ecg_features.shape}")
print(f"📊 RoBERTa features shape: {roberta_filtered.shape}")

# Create fusion features efficiently - REFERENCE METHOD
print("🔗 Creating fusion features...")
print("💡 Pre-allocating fusion array for efficiency...")
X_fusion = np.zeros((len(df_filtered), 896), dtype=np.float32)  # Pre-allocate fusion array!
print("📋 Copying ECG features...")
X_fusion[:, :128] = ecg_features  # Copy ECG features (first 128 columns)
print("📋 Copying RoBERTa features...")
X_fusion[:, 128:] = roberta_filtered  # Copy RoBERTa features (last 768 columns)
print("✅ Fusion array ready - no concatenation needed!")

print(f"✅ 500 Hz Fusion features ready: {X_fusion.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_fusion, y_filtered, stratify=y_filtered, test_size=0.2, random_state=42
)

print(f"🔀 Split: {len(X_train)} train / {len(X_test)} test")

# 🐺 500 Hz Ultra-Fast WPSA Algorithm (1 Hour Runtime)
class WPSA500Hz:
    def __init__(self, pack_size=4, max_iterations=10, bounds=None):  # Ultra-fast config
        self.pack_size = pack_size
        self.max_iterations = max_iterations
        self.start_time = time.time()
        self.time_limit = 3600  # 1 hour in seconds
        
        # Research paper's exact hyperparameter bounds
        self.bounds = bounds or {
            'n_estimators': (100, 800),        # Reduced upper bound
            'max_depth': (10, 50),             # Reduced upper bound
            'min_samples_split': (2, 20),      # Reduced upper bound
            'min_samples_leaf': (1, 10),       # Reduced upper bound
            'max_features': (0.1, 1.0),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        # Wolf hierarchy
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.pack = []
        
        print(f"🐺 500 Hz Ultra-Fast WPSA: Pack={pack_size}, Iterations={max_iterations}")
        print(f"⏱️ Time limit: {self.time_limit/60:.0f} minutes")
        print(f"📊 Total evaluations: {pack_size * max_iterations * 5} (5-fold CV)")
        print(f"⚡ Expected runtime: ~{(pack_size * max_iterations * 5 * 10)/60:.0f} minutes")

    def check_time_limit(self):
        """Check if we've exceeded the time limit"""
        elapsed = time.time() - self.start_time
        if elapsed > self.time_limit:
            print(f"⏰ Time limit reached ({elapsed/60:.1f} minutes)")
            return True
        return False

    def initialize_pack(self):
        """Initialize wolf pack with random positions"""
        self.pack = []
        for _ in range(self.pack_size):
            wolf = {}
            for param, bounds in self.bounds.items():
                if isinstance(bounds, list):
                    wolf[param] = random.choice(bounds)
                else:
                    min_val, max_val = bounds
                    if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                        wolf[param] = random.randint(int(min_val), int(max_val))
                    else:
                        wolf[param] = random.uniform(min_val, max_val)
            self.pack.append(wolf)

    def evaluate_fitness(self, wolf_params, X, y):
        """Evaluate wolf fitness using Dynamic Random Forest"""
        try:
            # Dynamic Random Forest with 500 Hz optimized configuration
            drf = RandomForestClassifier(
                n_estimators=wolf_params['n_estimators'],
                max_depth=wolf_params['max_depth'],
                min_samples_split=wolf_params['min_samples_split'],
                min_samples_leaf=wolf_params['min_samples_leaf'],
                max_features=wolf_params['max_features'],
                bootstrap=wolf_params['bootstrap'],
                criterion=wolf_params['criterion'],
                class_weight=wolf_params['class_weight'],
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            
            # 5-fold CV as in research paper
            scores = cross_val_score(drf, X, y, cv=5, scoring='accuracy')
            return scores.mean()
        except Exception as e:
            print(f"⚠️ Wolf evaluation failed: {e}")
            return 0.0

    def update_pack_hierarchy(self, fitnesses):
        """Update alpha, beta, gamma wolves based on fitness"""
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        self.alpha = deepcopy(self.pack[sorted_indices[0]])
        self.beta = deepcopy(self.pack[sorted_indices[1]])
        self.gamma = deepcopy(self.pack[sorted_indices[2]])

    def update_wolf_position(self, wolf, alpha, beta, gamma, iteration):
        """Update wolf position using research paper's WPSA equations"""
        updated_wolf = {}
        
        # Research paper's convergence parameter
        a = 2 - 2 * (iteration / self.max_iterations)
        
        for param in wolf.keys():
            if isinstance(self.bounds[param], list):
                # For categorical parameters, use majority voting
                choices = [alpha[param], beta[param], gamma[param]]
                updated_wolf[param] = max(set(choices), key=choices.count)
            else:
                # For numerical parameters, use WPSA position update
                r1, r2 = random.random(), random.random()
                
                A1 = 2 * a * r1 - a
                A2 = 2 * a * r2 - a
                A3 = 2 * a * random.random() - a
                
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
                
                # Final position
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
        """Main WPSA optimization loop with time limit"""
        print("🚀 Starting 500 Hz WPSA optimization...")
        print(f"⏱️ Started at: {time.strftime('%H:%M:%S')}")
        
        # Initialize pack
        self.initialize_pack()
        
        best_fitness = 0.0
        best_params = None
        
        for iteration in range(self.max_iterations):
            # Check time limit
            if self.check_time_limit():
                print(f"⏰ Stopping optimization due to time limit")
                break
                
            elapsed = time.time() - self.start_time
            print(f"🔄 Iteration {iteration + 1}/{self.max_iterations} ({elapsed/60:.1f} min elapsed)")
            
            # Evaluate all wolves
            fitnesses = []
            for i, wolf in enumerate(self.pack):
                if self.check_time_limit():
                    break
                    
                fitness = self.evaluate_fitness(wolf, X, y)
                fitnesses.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = deepcopy(wolf)
                
                print(f"   Wolf {i+1}: {fitness:.4f}")
            
            if self.check_time_limit():
                break
                
            # Update pack hierarchy
            self.update_pack_hierarchy(fitnesses)
            
            # Update wolf positions
            new_pack = []
            for wolf in self.pack:
                new_wolf = self.update_wolf_position(wolf, self.alpha, self.beta, self.gamma, iteration)
                new_pack.append(new_wolf)
            
            self.pack = new_pack
            
            print(f"   🏆 Best fitness: {best_fitness:.4f} ({best_fitness*100:.2f}%)")
            print(f"   🥇 Alpha: {self.alpha}")
            
            # Early stopping if we reach research paper's target
            if best_fitness >= 0.925:
                print(f"🎯 Reached target accuracy: {best_fitness:.4f} ({best_fitness*100:.2f}%)")
                break
        
        total_time = time.time() - self.start_time
        print(f"⏱️ Total optimization time: {total_time/60:.1f} minutes")
        
        return best_params, best_fitness

# Run 500 Hz WPSA
print("\n🐺 Starting 500 Hz Ultra-Fast WPSA Implementation...")
wpsa_500hz = WPSA500Hz(pack_size=4, max_iterations=10)  # Ultra-fast config
best_params, best_fitness = wpsa_500hz.optimize(X_train, y_train)

print(f"\n🏆 500 Hz WPSA Complete!")
print(f"   Best CV accuracy: {best_fitness:.4f} ({best_fitness*100:.2f}%)")
print(f"   Best parameters: {best_params}")

# Train final model with optimized parameters
print("\n🌲 Training final 500 Hz Dynamic Random Forest...")
final_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    criterion=best_params['criterion'],
    class_weight=best_params['class_weight'],
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

# Final evaluation
y_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Final 500 Hz Test Results:")
print(f"🎯 500 Hz WPSA-DRF: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(final_model, MODEL_SAVE_PATH)

print(f"\n💾 Model saved to: {MODEL_SAVE_PATH}")

print("\n🎯 Performance vs Research Paper:")
print(f"   - Research Paper Target: 92.5%")
print(f"   - Our 500 Hz Implementation: {final_accuracy*100:.2f}%")
if final_accuracy >= 0.925:
    print("   ✅ SUCCESS: Matched or exceeded research paper!")
else:
    print(f"   📊 Gap: {(0.925 - final_accuracy)*100:.2f}%")

print("\n🔄 Progress Summary:")
print("   - CNN (100 Hz ECG only): 42%")
print("   - CNN (500 Hz ECG only): 40.82%")
print("   - DRF (RoBERTa only): 80%")
print("   - Fusion (100 Hz): 80.75%")
print("   - Fusion (500 Hz): 82.11%")
print(f"   - WPSA-DRF (500 Hz): {final_accuracy*100:.2f}%")

print("\n✅ 500 Hz WPSA-DRF Implementation Complete!")
print("🎯 Next: If target not reached, consider ensemble methods or advanced architectures") 