import numpy as np

class MarkovSeyirClassifier:
    """Step 4: Use Markov Transition Matrices to identify the Maqam."""

    def __init__(self, bins_per_octave=24):
        self.bins_per_octave = bins_per_octave
        self.models = self._initialize_maqam_models()

    def _initialize_maqam_models(self, db_path="maqam_database.json"):
        import json
        import os
        
        library = {}

        # 1. Try to load from JSON
        if os.path.exists(db_path):
            try:
                with open(db_path, "r") as f:
                    data = json.load(f)
                # Convert lists back to numpy arrays
                for name, matrix_list in data.items():
                    library[name] = np.array(matrix_list)
                print(f"Loaded {len(library)} models from {db_path}")
                return library
            except Exception as e:
                print(f"Warning: Failed to load {db_path}: {e}")

        # 2. Fallback to hardcoded examples if no DB found
        print("Warning: No database found. Using hardcoded examples.")
        
        # Example: Bayati Transition Matrix
        bayati = np.full((self.bins_per_octave, self.bins_per_octave), 1e-6)
        bayati[0, 2] = 0.5  # D to E half-flat
        bayati[2, 10] = 0.3  # E half-flat to G
        library["Bayati"] = bayati / bayati.sum(axis=1)[:, None]

        # Example: Rast Transition Matrix
        rast = np.full((self.bins_per_octave, self.bins_per_octave), 1e-6)
        rast[0, 7] = 0.4  # C to E half-flat (index 7 in 24-bin)
        library["Rast"] = rast / rast.sum(axis=1)[:, None]
        return library

    def predict(self, sequence):
        results = {}
        for name, matrix in self.models.items():
            log_likelihood = 0
            for i in range(len(sequence) - 1):
                p = matrix[sequence[i], sequence[i + 1]]
                log_likelihood += np.log(p)
            results[name] = log_likelihood

        best_fit = max(results, key=results.get)
        return best_fit, results