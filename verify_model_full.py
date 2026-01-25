#!/usr/bin/env python3
"""
Comprehensive model verification script.
Infers class mapping by analyzing predictions across all maqams.
"""
import os
import numpy as np
import tensorflow as tf
import librosa
from collections import Counter, defaultdict

# Model and dataset paths
MODEL_PATH = '/Users/sezarhanna/Downloads/best_model_music_cnn_pitch.keras'
DATASET_PATH = '/Users/sezarhanna/Downloads/maqamidataset'

# Get maqam folders - sorted alphabetically (this is how Keras loads classes)
MAQAM_FOLDERS = sorted([d for d in os.listdir(DATASET_PATH) 
                        if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')])

print("=" * 70)
print("COMPREHENSIVE KERAS MODEL VERIFICATION")
print("=" * 70)

print(f"\nDataset folders (alphabetically sorted): {MAQAM_FOLDERS}")
print(f"Number of maqam folders: {len(MAQAM_FOLDERS)}")

# Standard alphabetical mapping for Keras ImageDataGenerator
# When flow_from_directory is used, classes are sorted alphabetically
ALPHABETICAL_CLASS_MAPPING = {i: name for i, name in enumerate(MAQAM_FOLDERS)}
print(f"\nAlphabetical class mapping (typical Keras default):")
for idx, name in ALPHABETICAL_CLASS_MAPPING.items():
    print(f"  Class {idx}: {name}")

def extract_features(audio_path, target_shape=(60, 358)):
    """Extract mel spectrogram features from audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=8.0)
        if len(y) < sr * 2:  # Skip very short files
            return None
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=60, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_spec_db.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :target_shape[1]]
        
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        return mel_spec_db
    except Exception as e:
        return None

# Load model
print("\n" + "-" * 70)
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded successfully!")
print(f"   Input shape: {model.input_shape}")
print(f"   Output classes: {model.output_shape[-1]}")

# Run inference on samples from each maqam folder
print("\n" + "-" * 70)
print("Running inference on dataset samples (10 per maqam)...")
print("-" * 70)

results = defaultdict(list)
samples_per_maqam = 10

for maqam in MAQAM_FOLDERS:
    maqam_path = os.path.join(DATASET_PATH, maqam)
    audio_files = [f for f in os.listdir(maqam_path) if f.endswith('.mp3')][:samples_per_maqam]
    
    for audio_file in audio_files:
        audio_path = os.path.join(maqam_path, audio_file)
        features = extract_features(audio_path)
        
        if features is not None:
            features = features.reshape(1, 60, 358, 1)
            pred = model.predict(features, verbose=0)
            pred_class = np.argmax(pred[0])
            confidence = pred[0][pred_class] * 100
            results[maqam].append((pred_class, confidence))

# Analyze results
print("\n" + "=" * 70)
print("PREDICTIONS ANALYSIS")
print("=" * 70)

class_to_maqams = defaultdict(list)

for maqam in MAQAM_FOLDERS:
    preds = [p[0] for p in results[maqam]]
    confs = [p[1] for p in results[maqam]]
    
    if preds:
        most_common = Counter(preds).most_common()
        main_class = most_common[0][0]
        main_count = most_common[0][1]
        avg_conf = np.mean(confs)
        
        class_to_maqams[main_class].append((maqam, main_count, len(preds)))
        
        print(f"\n{maqam}:")
        print(f"  Predictions: {preds}")
        print(f"  Distribution: {dict(Counter(preds))}")
        print(f"  Most predicted class: {main_class} ({main_count}/{len(preds)} = {100*main_count/len(preds):.0f}%)")
        print(f"  Avg confidence: {avg_conf:.1f}%")

# Infer class mapping
print("\n" + "=" * 70)
print("INFERRED CLASS MAPPING")
print("=" * 70)

print("\nBased on which class each maqam folder predominantly maps to:")
inferred_mapping = {}
for class_idx in sorted(class_to_maqams.keys()):
    maqams = class_to_maqams[class_idx]
    if len(maqams) == 1:
        maqam_name = maqams[0][0]
        inferred_mapping[class_idx] = maqam_name
        print(f"  Class {class_idx} → {maqam_name}")
    else:
        # Multiple maqams predict same class - pick the one with highest accuracy
        best = max(maqams, key=lambda x: x[1]/x[2])
        inferred_mapping[class_idx] = best[0]
        print(f"  Class {class_idx} → {best[0]} (contested: {[m[0] for m in maqams]})")

# Check for unmapped classes
all_classes = set(range(model.output_shape[-1]))
mapped_classes = set(inferred_mapping.keys())
unmapped = all_classes - mapped_classes

if unmapped:
    print(f"\n⚠️  Unmapped classes: {unmapped}")
    print("   These classes may correspond to maqams not in your dataset")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
✅ Model is TRAINED and produces predictions
• Model expects {model.output_shape[-1]} classes
• Dataset has {len(MAQAM_FOLDERS)} maqam folders: {MAQAM_FOLDERS}

Note: The model may have been trained with a different class order or 
preprocessing. If predictions don't match expected maqams, the model 
may need retraining on your specific dataset.

To use this model effectively:
1. Match your preprocessing to the model's expected input
2. Use the inferred class mapping above for predictions
3. Consider retraining on your dataset for better accuracy
""")
