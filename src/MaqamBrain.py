import numpy as np
import json
import os
from .MLPClassifier import MLPClassifier
from .JinsLibrary import JinsAnalyzer, MAQAM_STRUCTURE

class MaqamBrain:
    """
    The Central Intelligence of Maqam Detector 2.0.
    
    Hybrid Architecture with Jins-Based Detection:
    1. Jins Analyzer: Segments melody and identifies jins1 + jins2
    2. Markov Engine: Calculates transition log-likelihoods
    3. MLP Engine: Predicts probabilities from Seyir structure
    """
    
    def __init__(self, bins_per_octave=36, 
                 maqam_db_path="maqam_database_hybrid.json",
                 jins_db_path="jins_database.json",
                 mlp_path="mlp_model.pkl"):
        self.bins_per_octave = bins_per_octave
        self.markov_models = {}
        self.jins_models = {}
        self.mlp = MLPClassifier(model_path=mlp_path)
        self.jins_analyzer = JinsAnalyzer(bins_per_octave)
        
        self.maqam_db_path = maqam_db_path
        self.jins_db_path = jins_db_path
        
        self._load_markov_models()
        self._load_jins_models()

    def _load_markov_models(self):
        """Load maqam-level Markov transition matrices."""
        if os.path.exists(self.maqam_db_path):
            try:
                with open(self.maqam_db_path, "r") as f:
                    data = json.load(f)
                for name, matrix_list in data.items():
                    self.markov_models[name] = np.array(matrix_list)
                print(f"MaqamBrain: Loaded {len(self.markov_models)} Maqam Markov models.")
            except Exception as e:
                print(f"MaqamBrain: Failed to load Maqam Markov DB: {e}")
        else:
            print("MaqamBrain: No Maqam Markov DB found.")

    def _load_jins_models(self):
        """Load jins-level Markov transition matrices."""
        if os.path.exists(self.jins_db_path):
            try:
                with open(self.jins_db_path, "r") as f:
                    data = json.load(f)
                for name, matrix_list in data.items():
                    self.jins_models[name] = np.array(matrix_list)
                print(f"MaqamBrain: Loaded {len(self.jins_models)} Jins Markov models.")
            except Exception as e:
                print(f"MaqamBrain: Failed to load Jins Markov DB: {e}")
        else:
            print("MaqamBrain: No Jins Markov DB found.")

    def predict(self, sequence):
        """
        Returns hybrid prediction results with jins-level analysis.
        
        Returns:
            dict with:
            - prediction: Best maqam name
            - jins_analysis: Detected jins1 and jins2
            - markov_scores: Log-likelihoods per maqam
            - mlp_scores: MLP probabilities per maqam
            - confidence: Overall confidence score
        """
        # 1. Jins-based analysis (new!)
        jins_result = self.jins_analyzer.analyze_full_sequence(sequence)
        
        # 2. Markov predictions (maqam-level)
        markov_scores = {}
        if self.markov_models:
            for name, matrix in self.markov_models.items():
                log_likelihood = 0
                for i in range(len(sequence) - 1):
                    curr = min(int(sequence[i]), self.bins_per_octave - 1)
                    nxt = min(int(sequence[i+1]), self.bins_per_octave - 1)
                    p = matrix[curr, nxt]
                    log_likelihood += np.log(p + 1e-9)
                markov_scores[name] = log_likelihood
        
        # 3. MLP predictions
        seq_matrix = np.zeros((self.bins_per_octave, self.bins_per_octave))
        for i in range(len(sequence) - 1):
            curr = min(int(sequence[i]), self.bins_per_octave - 1)
            nxt = min(int(sequence[i+1]), self.bins_per_octave - 1)
            seq_matrix[curr, nxt] += 1
        
        row_sums = seq_matrix.sum(axis=1)[:, None]
        seq_matrix = np.divide(seq_matrix, row_sums, out=np.zeros_like(seq_matrix), where=row_sums!=0)
        
        mlp_scores = self.mlp.predict_proba(seq_matrix)
        
        # 4. Combine results using weighted voting
        best_maqam = self._combine_predictions(jins_result, markov_scores, mlp_scores)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(jins_result, markov_scores, mlp_scores, best_maqam)
            
        return {
            "prediction": best_maqam,
            "jins_analysis": {
                "jins1": jins_result.get("jins1", "Unknown"),
                "jins2": jins_result.get("jins2", "Unknown"),
                "jins_confidence": jins_result.get("confidence", 0.0),
                "jins2_root": jins_result.get("jins2_root", 15)
            },
            "markov_scores": markov_scores,
            "mlp_scores": mlp_scores,
            "confidence": confidence
        }
    
    def _combine_predictions(self, jins_result, markov_scores, mlp_scores):
        """
        Combine predictions from all three sources.
        Priority: MLP > Markov > Jins (if trained), else Jins > Markov
        """
        # If MLP is trained and has results, use it primarily
        if mlp_scores:
            return max(mlp_scores, key=mlp_scores.get)
        
        # If Markov models exist, use them
        if markov_scores:
            return max(markov_scores, key=markov_scores.get)
        
        # Fallback to jins-based prediction
        return jins_result.get("maqam", "Unknown")
    
    def _calculate_confidence(self, jins_result, markov_scores, mlp_scores, prediction):
        """Calculate overall confidence score (0-1)."""
        confidence_sources = []
        
        # Jins confidence
        if jins_result.get("confidence"):
            confidence_sources.append(jins_result["confidence"])
        
        # MLP confidence for the predicted class
        if mlp_scores and prediction in mlp_scores:
            confidence_sources.append(mlp_scores[prediction])
        
        # Markov confidence (normalized)
        if markov_scores and prediction in markov_scores:
            # Convert log-likelihood to relative confidence
            scores = list(markov_scores.values())
            if len(scores) > 1:
                score_range = max(scores) - min(scores)
                if score_range > 0:
                    markov_conf = (markov_scores[prediction] - min(scores)) / score_range
                    confidence_sources.append(markov_conf)
        
        if confidence_sources:
            return sum(confidence_sources) / len(confidence_sources)
        return 0.0

    def get_jins_details(self, maqam_name):
        """Get the jins structure for a specific maqam."""
        return MAQAM_STRUCTURE.get(maqam_name, {
            "jins1": "Unknown",
            "jins2": "Unknown",
            "jins1_root": 0,
            "jins2_root": 15
        })

    def predict_timeline(self, full_chromagram, window_seconds=10, hop_seconds=5, sr=22050):
        """
        Analyze audio in sliding windows to detect maqam changes over time.
        
        Args:
            full_chromagram: Complete chromagram of the audio
            window_seconds: Size of each analysis window in seconds
            hop_seconds: Hop size between windows in seconds
            sr: Sample rate used for chromagram
        
        Returns:
            dict with:
            - timeline: List of {start_time, end_time, maqam, jins1, jins2, confidence}
            - modulations: List of detected maqam changes
            - dominant_maqam: Most frequent maqam across all windows
        """
        # Calculate frame sizes (approximate based on librosa defaults)
        # librosa uses hop_length=512 by default, so each frame is ~23ms at 22050 Hz
        hop_length = 512
        frame_rate = sr / hop_length  # frames per second
        
        window_frames = int(window_seconds * frame_rate)
        hop_frames = int(hop_seconds * frame_rate)
        
        total_frames = full_chromagram.shape[1]
        
        timeline = []
        maqam_counts = {}
        
        # Slide window across the chromagram
        start_frame = 0
        while start_frame < total_frames:
            end_frame = min(start_frame + window_frames, total_frames)
            
            # Skip if window is too small
            if end_frame - start_frame < window_frames // 4:
                break
            
            # Extract window
            window_chroma = full_chromagram[:, start_frame:end_frame]
            
            # Convert to sequence (take max bin at each time step)
            sequence = np.argmax(window_chroma, axis=0)
            
            # Predict for this window
            result = self.predict(sequence)
            
            # Calculate timestamps
            start_time = start_frame / frame_rate
            end_time = end_frame / frame_rate
            
            maqam = result["prediction"]
            jins_analysis = result.get("jins_analysis", {})
            
            timeline.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "maqam": maqam,
                "jins1": jins_analysis.get("jins1", "Unknown"),
                "jins2": jins_analysis.get("jins2", "Unknown"),
                "confidence": round(result.get("confidence", 0), 3)
            })
            
            # Count maqam occurrences
            maqam_counts[maqam] = maqam_counts.get(maqam, 0) + 1
            
            start_frame += hop_frames
        
        # Detect modulations (maqam changes)
        modulations = []
        for i in range(1, len(timeline)):
            if timeline[i]["maqam"] != timeline[i-1]["maqam"]:
                modulations.append({
                    "time": timeline[i]["start_time"],
                    "from_maqam": timeline[i-1]["maqam"],
                    "to_maqam": timeline[i]["maqam"]
                })
        
        # Find dominant maqam
        dominant_maqam = max(maqam_counts, key=maqam_counts.get) if maqam_counts else "Unknown"
        
        return {
            "timeline": timeline,
            "modulations": modulations,
            "dominant_maqam": dominant_maqam,
            "total_segments": len(timeline),
            "window_seconds": window_seconds,
            "hop_seconds": hop_seconds
        }

