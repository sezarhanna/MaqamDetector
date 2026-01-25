import os
import numpy as np
import json

try:
    from .MLPClassifier import MLPClassifier
    from .JinsLibrary import MAQAM_STRUCTURE, JinsAnalyzer
except ImportError:
    from MLPClassifier import MLPClassifier
    from JinsLibrary import MAQAM_STRUCTURE, JinsAnalyzer

class MaqamTrainer:
    """
    Training Engine for Jins-Based MaqamDetector 2.0.
    
    Features:
    - 36-bin microtonal resolution
    - MP3 and WAV support
    - Jins-level transition learning (not just maqam-level)
    - Separate Markov models for jins1 and jins2
    """

    def __init__(self, bins_per_octave=36):
        self.bins_per_octave = bins_per_octave
        
        # Markov Counts: { "MaqamName": 36x36_matrix } for overall maqam
        self.markov_counts = {}
        
        # Jins-level Markov Counts: {"JinsName": 36x36_matrix}
        self.jins_counts = {}
        
        # MLP Training Data
        self.X_train = []  # List of flattened transition matrices
        self.y_train = []  # List of labels
        
        # Jins analyzer for segmentation
        self.jins_analyzer = JinsAnalyzer(bins_per_octave)
        
        # Supported audio formats
        self.audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    def train_on_folder(self, maqam_name, folder_path, processor, finder, normalizer):
        """Processes all audio files in a folder."""
        if maqam_name not in self.markov_counts:
            self.markov_counts[maqam_name] = np.zeros((self.bins_per_octave, self.bins_per_octave))
        
        # Get jins structure for this maqam
        maqam_struct = MAQAM_STRUCTURE.get(maqam_name, None)
        if maqam_struct:
            jins1_name = maqam_struct["jins1"]
            jins2_name = maqam_struct["jins2"]
            jins2_root = maqam_struct.get("jins2_root", 15)
            
            if jins1_name not in self.jins_counts:
                self.jins_counts[jins1_name] = np.zeros((self.bins_per_octave, self.bins_per_octave))
            if jins2_name not in self.jins_counts:
                self.jins_counts[jins2_name] = np.zeros((self.bins_per_octave, self.bins_per_octave))
        else:
            jins1_name = jins2_name = None
            jins2_root = 15

        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} not found.")
            return

        print(f"Training: Processing {maqam_name}...")
        files_processed = 0
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(self.audio_extensions):
                file_path = os.path.join(folder_path, filename)

                try:
                    # 1. Signal Processing
                    chroma = processor.get_chromagram(file_path)
                    if chroma.shape[1] < 2: 
                        continue  # Skip empty/short
                    
                    # 2. Find Rukooz
                    rukooz = finder.find_rukooz(chroma)
                    
                    # 3. Normalize Sequence
                    sequence = normalizer.normalize(chroma, rukooz)
                    
                    # 4. Update Maqam-level Markov Counts
                    file_matrix = np.zeros((self.bins_per_octave, self.bins_per_octave))
                    
                    for i in range(len(sequence) - 1):
                        idx_curr = int(sequence[i]) % self.bins_per_octave
                        idx_next = int(sequence[i + 1]) % self.bins_per_octave
                        
                        self.markov_counts[maqam_name][idx_curr, idx_next] += 1
                        file_matrix[idx_curr, idx_next] += 1
                    
                    # 5. Update Jins-level Markov Counts
                    if jins1_name and jins2_name:
                        jins1_seq, jins2_seq = self.jins_analyzer.segment_into_jins(sequence, jins2_root)
                        
                        # Learn jins1 transitions
                        for i in range(len(jins1_seq) - 1):
                            idx_curr = int(jins1_seq[i]) % self.bins_per_octave
                            idx_next = int(jins1_seq[i + 1]) % self.bins_per_octave
                            self.jins_counts[jins1_name][idx_curr, idx_next] += 1
                        
                        # Learn jins2 transitions (already normalized relative to jins2 root)
                        for i in range(len(jins2_seq) - 1):
                            idx_curr = int(jins2_seq[i]) % self.bins_per_octave
                            idx_next = int(jins2_seq[i + 1]) % self.bins_per_octave
                            self.jins_counts[jins2_name][idx_curr, idx_next] += 1
                    
                    # 6. Prepare MLP Data
                    row_sums = file_matrix.sum(axis=1)[:, None]
                    file_probs = np.divide(file_matrix, row_sums, out=np.zeros_like(file_matrix), where=row_sums!=0)
                    
                    self.X_train.append(file_probs.flatten())
                    self.y_train.append(maqam_name)
                    
                    files_processed += 1
                    
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
        
        print(f"  Processed {files_processed} files for {maqam_name}")

    def finalize_and_save(self, markov_file="maqam_database_hybrid.json", 
                          jins_file="jins_database.json",
                          mlp_file="mlp_model.pkl"):
        """Saves Markov models (maqam + jins levels) and trains/saves the MLP model."""
        
        # 1. Save Maqam-level Markov Models
        trained_library = {}
        for name, matrix in self.markov_counts.items():
            # Add smoothing and normalize
            smooth_matrix = matrix + 1e-6
            prob_matrix = smooth_matrix / smooth_matrix.sum(axis=1)[:, None]
            trained_library[name] = prob_matrix.tolist()

        with open(markov_file, 'w') as f:
            json.dump(trained_library, f, indent=4)
        print(f"Success: Maqam Markov models saved to {markov_file}")
        
        # 2. Save Jins-level Markov Models
        if self.jins_counts:
            jins_library = {}
            for name, matrix in self.jins_counts.items():
                smooth_matrix = matrix + 1e-6
                prob_matrix = smooth_matrix / smooth_matrix.sum(axis=1)[:, None]
                jins_library[name] = prob_matrix.tolist()
            
            with open(jins_file, 'w') as f:
                json.dump(jins_library, f, indent=4)
            print(f"Success: Jins Markov models saved to {jins_file}")

        # 3. Train and Save MLP
        if self.X_train:
            print(f"Training MLP on {len(self.X_train)} samples...")
            mlp = MLPClassifier(model_path=mlp_file)
            mlp.train(self.X_train, self.y_train)
        else:
            print("Warning: No data for MLP training.")