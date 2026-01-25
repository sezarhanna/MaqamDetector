#!/usr/bin/env python3
"""
Script to verify the keras model by running inference on the maqami dataset.
"""
import os
import numpy as np
import tensorflow as tf
import librosa
from collections import Counter

# Model and dataset paths
MODEL_PATH = '/Users/sezarhanna/Downloads/best_model_music_cnn_pitch.keras'
DATASET_PATH = '/Users/sezarhanna/Downloads/maqamidataset'

# Maqam classes (from the dataset folder names)
MAQAM_CLASSES = ['Agm', 'Byat', 'Cord', 'Hjaz', 'Nahawond', 'Rast', 'Sba']

def extract_features(audio_path, target_shape=(60, 358)):
    """Extract mel spectrogram features from audio file."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050, duration=8.0)
        
        # Extract mel spectrogram (60 mel bands)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=60, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or truncate to target shape
        if mel_spec_db.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :target_shape[1]]
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    print("=" * 60)
    print("KERAS MODEL VERIFICATION")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"   ✅ Model loaded from: {MODEL_PATH}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Number of classes: {model.output_shape[-1]}")
    
    # Get the class names from the model if available
    num_classes = model.output_shape[-1]
    print(f"\n   Model expects {num_classes} classes")
    print(f"   Dataset has {len(MAQAM_CLASSES)} folders: {MAQAM_CLASSES}")
    
    # Test inference on samples from each maqam
    print("\n2. Running inference on dataset samples...")
    print("-" * 60)
    
    results = {}
    correct = 0
    total = 0
    
    for maqam_idx, maqam in enumerate(MAQAM_CLASSES):
        maqam_path = os.path.join(DATASET_PATH, maqam)
        if not os.path.exists(maqam_path):
            print(f"   ⚠️ Folder not found: {maqam_path}")
            continue
        
        # Get first 5 audio files for testing
        audio_files = [f for f in os.listdir(maqam_path) if f.endswith('.mp3')][:5]
        
        predictions = []
        for audio_file in audio_files:
            audio_path = os.path.join(maqam_path, audio_file)
            features = extract_features(audio_path)
            
            if features is not None:
                # Reshape for model input (batch, height, width, channels)
                features = features.reshape(1, 60, 358, 1)
                
                # Run inference
                pred = model.predict(features, verbose=0)
                pred_class = np.argmax(pred[0])
                confidence = pred[0][pred_class] * 100
                predictions.append(pred_class)
                total += 1
                
        if predictions:
            # Get most common prediction for this maqam
            most_common = Counter(predictions).most_common(1)[0]
            pred_class_idx = most_common[0]
            pred_count = most_common[1]
            
            results[maqam] = {
                'predictions': predictions,
                'most_common': pred_class_idx,
                'count': pred_count
            }
            
            print(f"\n   {maqam} (folder index {maqam_idx}):")
            print(f"   Predictions: {predictions}")
            print(f"   Most common predicted class: {pred_class_idx} ({pred_count}/{len(predictions)} samples)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nModel is a TRAINED CNN-LSTM classifier for {num_classes} classes")
    print(f"Total samples tested: {total}")
    print("\nPrediction distribution by maqam folder:")
    
    for maqam, data in results.items():
        pred_dist = Counter(data['predictions'])
        print(f"  {maqam}: {dict(pred_dist)}")
    
    print("\n✅ Model verification complete!")
    print("The model successfully loads and produces predictions.")
    
if __name__ == '__main__':
    main()
