import numpy as np
import os
import librosa
import tensorflow as tf
from MarkovSeyirClassifier import MarkovSeyirClassifier
from SignalProcessor import SignalProcessor
from TonicFinder import TonicFinder
from SequenceNormalizer import SequenceNormalizer

# Maqam mapping based on the Kaggle notebook order
MAQAM_MAPPING = {
    0: "Ajam",      # Agm
    1: "Hijaz",     # Hjaz
    2: "Bayati",    # Byat
    3: "Nahawand",  # Nahawond
    4: "Saba",      # Sba
    5: "Sikah",     # Sekah
    6: "Rast",      # Rast
    7: "Kurd"       # Cord
}

class MaqamBrain:
    """
    The Central Intelligence of Maqam Detector.
    Now powered by a pre-trained Keras CNN model.
    """
    
    def __init__(self, model_path, bins_per_octave=36):
        self.model_path = model_path
        self.model = None
        self.output_classes = 8
        self._load_model()

        # Initialize Markov Brain components
        self.processor = SignalProcessor()
        self.finder = TonicFinder()
        self.normalizer = SequenceNormalizer()
        self.markov_brain = MarkovSeyirClassifier()

    def _load_model(self):
        """Load the Keras model."""
        if os.path.exists(self.model_path):
            try:
                print(f"MaqamBrain: Loading model from {self.model_path}...")
                self.model = tf.keras.models.load_model(self.model_path)
                print("MaqamBrain: Model loaded successfully!")
                self.output_classes = self.model.output_shape[-1]
            except Exception as e:
                print(f"MaqamBrain: Failed to load model: {e}")
        else:
            print(f"MaqamBrain: Model not found at {self.model_path}")

    def predict_file(self, file_path, algorithm="cnn"):
        """
        Predict maqam from an audio file.
        Algorithm: 'cnn' or 'markov'.
        """
        if algorithm == "markov":
            return self.predict_markov_file(file_path)
            
        # Default to CNN
        if self.model is None:
            return {"prediction": "Model Error", "confidence": 0.0, "details": "Model not loaded"}

        try:
            # 1. Preprocess
            features = self._extract_features(file_path)
            if features is None:
                 return {"prediction": "Processing Error", "confidence": 0.0, "details": "Could not extract features"}
            
            # 2. Reshape for CNN (Batch, Mels, Time, Channels)
            # Model expects (None, 60, 358, 1) based on analysis
            features = features.reshape(1, 60, 358, 1)
            
            # 3. Predict
            pred_probs = self.model.predict(features, verbose=0)[0]
            pred_class_idx = np.argmax(pred_probs)
            confidence = float(pred_probs[pred_class_idx])
            
            predicted_maqam = MAQAM_MAPPING.get(pred_class_idx, f"Unknown ({pred_class_idx})")
            
            # Create scores dict
            scores = {MAQAM_MAPPING.get(i, str(i)): float(prob) for i, prob in enumerate(pred_probs)}
            import json
            print(f"Prediction: {predicted_maqam} ({confidence:.2f})")
            print(f"Scores: {json.dumps(scores, indent=2)}")
            
            return {
                "prediction": predicted_maqam,
                "confidence": confidence,
                "scores": scores,
                "jins_analysis": {"jins1": "N/A", "jins2": "N/A"} # Legacy support
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"prediction": "Error", "confidence": 0.0, "details": str(e)}

    def _extract_features(self, audio_path, target_shape=(60, 358)):
        """
        Extract Mel Spectrogram features matching the training:
        - sr=22050
        - duration=8.0s (approx)
        - n_mels=60
        - n_fft=2048
        - hop_length=512
        """
        try:
            # Load audio, ensure mono
            y, sr = librosa.load(audio_path, sr=22050, duration=8.0)
            
            # Pad if too short (at least 1 second)
            if len(y) < sr:
                y = np.pad(y, (0, sr - len(y)))
                
            # Compute Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=60, 
                n_fft=2048, 
                hop_length=512
            )
            
            # Convert to Log Scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Fix Time Dimension to 358
            current_width = mel_spec_db.shape[1]
            target_width = target_shape[1]
            
            if current_width < target_width:
                # Pad with zeros
                pad_width = target_width - current_width
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Crop
                mel_spec_db = mel_spec_db[:, :target_width]
                
            # Normalize (Min-Max to 0-1 range)
            # This is critical if the model was trained with normalized data
            min_val = mel_spec_db.min()
            max_val = mel_spec_db.max()
            if max_val - min_val > 0:
                mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val)
            else:
                mel_spec_db = np.zeros_like(mel_spec_db)
            
            print(f"Spectrogram Stats - Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mel_spec_db.mean():.2f}")
                
            return mel_spec_db
            
        except Exception as e:
            import os
            try:
                size = os.path.getsize(audio_path)
            except:
                size = "unknown"
            print(f"Feature extraction failed for {audio_path} (size={size}): {repr(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def predict_markov_file(self, file_path):
        """
        Predict using the Markov Chain logic (classic method).
        """
        try:
            # 1. Signal Processing
            chroma = self.processor.get_chromagram(file_path)
            if chroma.shape[1] < 2:
                return {"prediction": "Error", "confidence": 0.0, "details": "Audio too short"}
            
            # 2. Find Rukooz
            rukooz = self.finder.find_rukooz(chroma)
            
            # 3. Normalize Sequence
            sequence = self.normalizer.normalize(chroma, rukooz)
            
            # 4. Predict using Markov
            best_fit, results = self.markov_brain.predict(sequence)
            
            # Normalize scores for frontend
            # Softmax-ish or just normalize log likelihoods? 
            # Log likelihoods are negative. Max is best.
            # Simple approach: Return raw logs or a relative score
            
            return {
                "prediction": best_fit,
                "confidence": 1.0, # Placeholder, Markov doesn't give probability easily
                "scores": results,
                "details": "Markov Prediction"
            }
            
        except Exception as e:
            print(f"Markov Prediction Error: {e}")
            return {"prediction": "Error", "confidence": 0.0, "details": str(e)}

    # Legacy method support (stubbed)
    def predict(self, sequence):
        return {"prediction": "Use predict_file instead", "confidence": 0.0}
    
    def predict_timeline(self, full_chromagram, window_seconds=10, hop_seconds=5, sr=22050):
         pass # Timeline not supported with this CNN model yet
