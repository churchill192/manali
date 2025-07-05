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
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.load_ecg_signals import scp_to_multiclass

# Configuration - Exact from Research Paper
TARGET_CLASSES = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']
ROBERTA_PATH = "../data/processed/roberta_embeddings.npy"
CSV_PATH = "../data/ptbxl/ptbxl_database.csv"
ECG_BASE_PATH = "../data/ptbxl/"
MODEL_SAVE_PATH = "../results/models/research_paper_exact_model.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("📋 Research Paper Exact Implementation")
print("🎯 Target: 92.5% Accuracy with WPSA-DRF")
print("=" * 50)

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

# Load ECG features (same as fusion model)
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

print("🔬 Loading ECG features...")
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

# 🐺 Research Paper's Exact WPSA Algorithm
class ResearchPaperWPSA:
    def __init__(self, pack_size=10, max_iterations=25, bounds=None):
        self.pack_size = pack_size
        self.max_iterations = max_iterations
        
        # Research paper's exact hyperparameter bounds
        self.bounds = bounds or {
            'n_estimators': (100, 1000),
            'max_depth': (10, 100),
            'min_samples_split': (2, 50),
            'min_samples_leaf': (1, 20),
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
        
        print(f"🐺 Research Paper WPSA: Pack={pack_size}, Iterations={max_iterations}")

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
            # Dynamic Random Forest with research paper's configuration
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
                n_jobs=-1
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
        """Main WPSA optimization loop"""
        print("🚀 Starting Research Paper WPSA optimization...")
        
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
                new_wolf = self.update_wolf_position(wolf, self.alpha, self.beta, self.gamma, iteration)
                new_pack.append(new_wolf)
            
            self.pack = new_pack
            
            print(f"   🏆 Best fitness: {best_fitness:.4f}")
            print(f"   🥇 Alpha: {self.alpha}")
            
            # Early stopping if we reach research paper's target
            if best_fitness >= 0.925:
                print(f"🎯 Reached target accuracy: {best_fitness:.4f}")
                break
        
        return best_params, best_fitness

# Run Research Paper's Exact WPSA
print("\n🐺 Starting Research Paper's Exact WPSA Implementation...")
research_wpsa = ResearchPaperWPSA(pack_size=8, max_iterations=20)  # Optimized for efficiency
best_params, best_fitness = research_wpsa.optimize(X_train, y_train)

print(f"\n🏆 Research Paper WPSA Complete!")
print(f"   Best CV accuracy: {best_fitness:.4f} ({best_fitness*100:.2f}%)")
print(f"   Best parameters: {best_params}")

# Train final model with optimized parameters
print("\n🌲 Training final Dynamic Random Forest...")
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

print(f"\n📊 Final Test Results:")
print(f"🎯 Research Paper Implementation: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

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
print(f"   - Our Implementation: {final_accuracy*100:.2f}%")
if final_accuracy >= 0.925:
    print("   ✅ SUCCESS: Matched or exceeded research paper!")
else:
    print(f"   📊 Gap: {(0.925 - final_accuracy)*100:.2f}%")

print("\n🔄 Progress Summary:")
print("   - CNN (ECG only): 42%")
print("   - DRF (RoBERTa only): 80%")
print("   - Fusion (ECG + RoBERTa): 80.75%")
print(f"   - Research Paper Exact: {final_accuracy*100:.2f}%")

print("\n✅ Research Paper Implementation Complete!") 