import numpy as np

class MarkovSeyirClassifier:
    """
    Step 4: Use Markov Transition Matrices to identify the Maqam.
    
    NOTE: This is a legacy classifier. The new MaqamBrain uses JinsAnalyzer
    for more accurate jins-based detection.
    """

    def __init__(self, bins_per_octave=36):
        self.bins_per_octave = bins_per_octave
        self.models = self._initialize_maqam_models()

    def _initialize_maqam_models(self, db_path="maqam_database_hybrid.json"):
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

        # 2. Fallback to hardcoded examples if no DB found (36-bin)
        print("Warning: No database found. Using hardcoded 36-bin examples.")
        
        # Example: Bayati Transition Matrix (36-bin)
        # Bayati on D: D(0) - E♭neutral(5) - F(9) - G(15)
        bayati = np.full((self.bins_per_octave, self.bins_per_octave), 1e-6)
        bayati[0, 5] = 0.4   # D to E♭ (neutral 2nd)
        bayati[5, 9] = 0.3   # E♭ to F
        bayati[9, 15] = 0.2  # F to G (4th)
        bayati[15, 9] = 0.3  # G to F (descending)
        library["Bayati"] = bayati / bayati.sum(axis=1)[:, None]

        # Example: Rast Transition Matrix (36-bin)
        # Rast on C: C(0) - D(6) - E♭neutral(10) - F(15)
        rast = np.full((self.bins_per_octave, self.bins_per_octave), 1e-6)
        rast[0, 6] = 0.4    # C to D
        rast[6, 10] = 0.3   # D to E♭ neutral
        rast[10, 15] = 0.2  # E♭ to F
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