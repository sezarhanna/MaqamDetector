import os
import numpy as np
import json


class MaqamTrainer:
    """Step 5: Training Engine to build Markov Matrices from your own Oud recordings."""

    def __init__(self, bins_per_octave=24):
        self.bins_per_octave = bins_per_octave
        # Stores raw counts of transitions: { "MaqamName": 24x24_matrix }
        self.counts = {}

    def train_on_folder(self, maqam_name, folder_path, processor, finder, normalizer):
        """Processes all .wav files in a folder to learn a specific Maqam's Seyir."""
        if maqam_name not in self.counts:
            self.counts[maqam_name] = np.zeros((self.bins_per_octave, self.bins_per_octave))

        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} not found.")
            return

        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                print(f"Processing training file: {filename}...")

                # Extract the normalized pitch sequence (Steps 1, 2, and 3)
                chroma = processor.get_chromagram(file_path)
                rukooz = finder.find_rukooz(chroma)
                sequence = normalizer.normalize(chroma, rukooz)

                # Update the Markov transition counts
                for i in range(len(sequence) - 1):
                    current_note = sequence[i]
                    next_note = sequence[i + 1]
                    self.counts[maqam_name][current_note, next_note] += 1

    def finalize_and_save(self, output_file="maqam_database.json"):
        """Normalizes counts into probabilities and saves the model to JSON."""
        trained_library = {}

        for name, matrix in self.counts.items():
            # Add Laplace Smoothing (epsilon) to prevent zero-probability errors in log-likelihood
            smooth_matrix = matrix + 1e-6
            # Normalize rows to sum to 1.0
            prob_matrix = smooth_matrix / smooth_matrix.sum(axis=1)[:, None]

            # Convert numpy array to list for JSON serialization
            trained_library[name] = prob_matrix.tolist()

        with open(output_file, 'w') as f:
            json.dump(trained_library, f, indent=4)

        print(f"Success: Trained models saved to {output_file}")
        return trained_library