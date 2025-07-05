import numpy as np
from preprocessing.load_ecg_signals import (
    load_metadata,
    build_dataset,
    scp_to_multiclass,
    filter_df_by_target_classes
)

if __name__ == "__main__":
    df = load_metadata()
    
    # Use actual diagnostic codes from PTB-XL dataset
    target_classes = ['NORM', 'IMI', 'ASMI', 'LVH', 'LAFB', 'AFIB']
    df_balanced = filter_df_by_target_classes(df, target_classes, samples_per_class=250)

    X, y_raw = build_dataset(df_balanced)
    y_labels = scp_to_multiclass(y_raw, target_classes)

    print("✅ Final multiclass target shape:", y_labels.shape)
    print("🧾 Sample multiclass label:", y_labels[0])
    print("📊 Unique classes:", np.unique(y_labels))
    
    # Show class distribution
    unique, counts = np.unique(y_labels, return_counts=True)
    print("📈 Class distribution:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        if cls >= 0:
            print(f"   {target_classes[cls]} ({cls}): {count} samples")
        else:
            print(f"   Invalid (-1): {count} samples")
            
    # Show what each class represents
    class_descriptions = {
        'NORM': 'Normal ECG',
        'IMI': 'Inferior Myocardial Infarction',
        'ASMI': 'Anteroseptal Myocardial Infarction', 
        'LVH': 'Left Ventricular Hypertrophy',
        'LAFB': 'Left Anterior Fascicular Block',
        'AFIB': 'Atrial Fibrillation'
    }
    
    print("\n🏥 Cardiovascular Condition Classes:")
    for i, cls in enumerate(target_classes):
        print(f"   {i}: {cls} - {class_descriptions[cls]}")
