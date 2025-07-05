import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import os
import torch
from transformers import AutoTokenizer, AutoModel

# Settings
CSV_PATH = "../data/ptbxl/ptbxl_database.csv"
SAVE_PATH = "../data/processed/roberta_embeddings.npy"
MODEL_NAME = "roberta-base"  # same family as paper used
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

print(f"📦 Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def parse_scp_to_sentence(scp):
    try:
        code_dict = ast.literal_eval(scp)
        if not code_dict:
            return "no significant findings"
        codes = sorted(code_dict.items(), key=lambda x: -x[1])
        phrases = [code_mapping.get(code[0], code[0]) for code in codes[:3]]
        return "ECG shows " + ", ".join(phrases)
    except:
        return "no diagnosis found"

# Map SCP codes to readable text
code_mapping = {
    "NORM": "normal sinus rhythm",
    "AFIB": "atrial fibrillation",
    "IMI": "inferior myocardial infarction",
    "ASMI": "anteroseptal myocardial infarction",
    "LVH": "left ventricular hypertrophy",
    "LAFB": "left anterior fascicular block",
    "RBBB": "right bundle branch block",
    "PVC": "premature ventricular contractions",
    "SBRAD": "sinus bradycardia",
    "TINV": "T wave inversion"
}

def generate_sentences(df):
    print("📝 Generating clinical sentences from SCP codes or reports...")
    texts = []

    for _, row in df.iterrows():
        if pd.notna(row["report"]) and isinstance(row["report"], str) and len(row["report"]) > 5:
            texts.append(row["report"].strip())
        else:
            texts.append(parse_scp_to_sentence(row["scp_codes"]))
    
    return texts

def generate_embeddings(texts, batch_size=16):
    print(f"🔤 Generating RoBERTa embeddings for {len(texts)} texts...")
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output = model(**encoded)
            cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)

def main():
    df = pd.read_csv(CSV_PATH)

    # Filter only ECGs with selected diagnostic classes
    target_classes = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']
    df = df[df['scp_codes'].notna()]
    df = df[df['scp_codes'].str.contains('|'.join(target_classes))]
    df = df.drop_duplicates(subset="ecg_id")

    # Generate text
    texts = generate_sentences(df)

    # Embed
    embeddings = generate_embeddings(texts)
    print(f"✅ Final shape: {embeddings.shape}")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.save(SAVE_PATH, embeddings)
    print(f"💾 Saved embeddings to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
