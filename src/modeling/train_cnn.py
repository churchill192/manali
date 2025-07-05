import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import sys

# Add parent directory to path so we can import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load preprocessed data
from preprocessing.load_ecg_signals import load_metadata, filter_df_by_target_classes, build_dataset, scp_to_multiclass

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 6

# CNN Model
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.flatten = nn.Flatten()
        # Fixed: After 2 pooling layers: 1000 -> 500 -> 250, so 64 * 250 = 16000
        self.fc1 = nn.Linear(64 * 250, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # (B, 32, 500)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # (B, 64, 250)
        x = self.flatten(x)  # (B, 64*250 = 16000)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# Dataset class
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Train function
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in tqdm(dataloader, desc="🔁 Training"):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Eval function
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    return np.array(all_preds), np.array(all_targets)

# Main
if __name__ == "__main__":
    print("🏥 Starting CNN Baseline Training for Cardiovascular Classification")
    print("=" * 60)
    
    df = load_metadata()
    target_classes = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']
    df_balanced = filter_df_by_target_classes(df, target_classes, samples_per_class=250)
    X, y_raw = build_dataset(df_balanced)
    y_labels = scp_to_multiclass(y_raw, target_classes)

    # Filter out -1 labels
    mask = y_labels != -1
    X = X[mask]
    y_labels = y_labels[mask]
    
    print(f"📊 Final dataset: {X.shape[0]} samples with {len(np.unique(y_labels))} classes")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    
    print(f"🔄 Train/Test split: {len(X_train)} train, {len(X_test)} test samples")

    # Dataloaders
    train_ds = ECGDataset(X_train, y_train)
    test_ds = ECGDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = ECGClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"🤖 Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Training loop
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        loss = train(model, train_dl, criterion, optimizer)
        print(f"📉 Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

    # Evaluation
    print("\n📊 Evaluating model on test set...")
    preds, targets = evaluate(model, test_dl)
    
    print("\n📊 Classification Report:")
    print(classification_report(targets, preds, target_names=target_classes))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(targets, preds))

    # Save model
    os.makedirs("../results/models", exist_ok=True)
    torch.save(model.state_dict(), "../results/models/cnn_ecg_model.pth")
    print("💾 Model saved to ../results/models/cnn_ecg_model.pth")
    
    print("\n✅ CNN Baseline Training Complete!")
    print("🎯 Next: Compare results with research paper's WPSA-DRF method")
