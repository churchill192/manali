import pandas as pd
import numpy as np
import wfdb
import os
from tqdm import tqdm
import ast

def load_metadata(path="../data/ptbxl/ptbxl_database.csv"):
    """
    Load and analyze PTB-XL cardiovascular dataset
    """
    df = pd.read_csv(path)
    print("✅ PTB-XL Cardiovascular Dataset loaded:", df.shape)
    print(f"📊 Total patients: {df['patient_id'].nunique()}")
    print(f"📊 Total ECG recordings: {len(df)}")
    
    # Show basic patient demographics
    print("\n👥 Patient Demographics:")
    print(f"   Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    print(f"   Gender distribution: {df['sex'].value_counts().to_dict()}")
    
    # Show diagnostic information
    print("\n🔍 Diagnostic Information:")
    print("   Sample diagnostic codes (scp_codes):")
    print(df[['ecg_id', 'scp_codes']].head())
    
    # Show clinical reports (perfect for NLP)
    print("\n📋 Clinical Reports (for NLP processing):")
    print("   Sample reports:")
    non_empty_reports = df[df['report'].notna()]
    if len(non_empty_reports) > 0:
        print(non_empty_reports[['ecg_id', 'report']].head(2))
    else:
        print("   No text reports found")
    
    # Show ECG file information
    print("\n📡 ECG Signal Files:")
    print(f"   High-resolution files: {df['filename_hr'].notna().sum()}")
    print(f"   Low-resolution files: {df['filename_lr'].notna().sum()}")
    
    return df

def filter_df_by_target_classes(df, target_classes, samples_per_class=300):
    """
    Return a balanced DataFrame with ECGs from each target class.
    """
    selected_rows = []
    
    print(f"🔍 Searching for balanced samples across {len(target_classes)} classes...")
    
    for cls in target_classes:
        # Filter rows that contain this class in their scp_codes
        matched = df[df['scp_codes'].str.contains(cls, na=False)]
        
        if len(matched) > 0:
            sampled = matched.sample(n=min(samples_per_class, len(matched)), random_state=42)
            selected_rows.append(sampled)
            print(f"   ✅ {cls}: Found {len(matched)} cases, sampled {len(sampled)}")
        else:
            print(f"   ❌ {cls}: No cases found")
    
    if selected_rows:
        balanced_df = pd.concat(selected_rows).drop_duplicates().reset_index(drop=True)
        print(f"✅ Balanced sample: {len(balanced_df)} ECGs across {len(target_classes)} classes")
        return balanced_df
    else:
        print("❌ No balanced samples found, returning original sample")
        return df.head(500)

def load_ecg_signal(record_path, base_dir="../data/ptbxl/"):
    """
    Loads a single ECG recording using wfdb.
    Returns a NumPy array of shape (12, 1000) for 12 leads, 10 seconds at 100 Hz.
    """
    full_path = os.path.join(base_dir, record_path.replace('.dat', ''))
    try:
        record = wfdb.rdrecord(full_path)
        return record.p_signal.T  # Transpose: (1000, 12) → (12, 1000)
    except Exception as e:
        print(f"❌ Failed to load: {full_path} — {e}")
        return None

def build_dataset(df, base_dir="../data/ptbxl/", leads=12, length=1000):
    """
    Build ECG signal dataset from PTB-XL filenames in DataFrame.
    """
    signals = []
    ids = []
    
    print(f"📥 Building ECG signal dataset from {len(df)} records...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        signal = load_ecg_signal(row['filename_lr'], base_dir)
        if signal is not None and signal.shape == (leads, length):
            signals.append(signal)
            ids.append(row['ecg_id'])
    
    X = np.stack(signals)  # Shape: (n_samples, 12, 1000)
    y = df.loc[df['ecg_id'].isin(ids), 'scp_codes'].values
    print(f"✅ Loaded {X.shape[0]} ECGs, shape: {X.shape}")
    
    return X, y

def scp_to_multiclass(y_raw, target_classes=None):
    """
    Convert list of SCP code dicts into multiclass labels.
    If multiple labels match, choose the one with highest score.
    """
    if target_classes is None:
        target_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'AFIB']
    
    label_indices = []
    
    for code_dict in y_raw:
        if isinstance(code_dict, str):
            try:
                code_dict = ast.literal_eval(code_dict)
            except:
                label_indices.append(-1)
                continue

        # Filter only target labels
        filtered = {k: v for k, v in code_dict.items() if k in target_classes}
        if filtered:
            # Pick the class with highest value
            best_label = max(filtered, key=filtered.get)
            label_indices.append(target_classes.index(best_label))
        else:
            label_indices.append(-1)  # Invalid or no match
    
    return np.array(label_indices)

def analyze_diagnostic_codes(df):
    """
    Analyze diagnostic codes for cardiovascular conditions
    """
    print("\n🔬 Analyzing Cardiovascular Diagnostic Codes...")
    
    # Parse diagnostic codes
    diagnostic_codes = []
    for codes in df['scp_codes'].dropna():
        try:
            code_dict = ast.literal_eval(codes)
            diagnostic_codes.extend(code_dict.keys())
        except:
            continue
    
    # Show most common diagnostic codes
    from collections import Counter
    code_counts = Counter(diagnostic_codes)
    print(f"\n📈 Top 10 Most Common Diagnostic Codes:")
    for code, count in code_counts.most_common(10):
        print(f"   {code}: {count} cases")
    
    return code_counts

if __name__ == "__main__":
    df = load_metadata()
    analyze_diagnostic_codes(df)
