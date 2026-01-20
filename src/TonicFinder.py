import numpy as np

class TonicFinder:
    """Step 2: Analysis of energy distribution to find the Rukooz (Tonic)."""
    def __init__(self, bins_per_octave=36):
        self.bins_per_octave = bins_per_octave

    def find_rukooz(self, chromagram):
        """
        Identifies the tonic bin index (0-35).
        """
        # Sum energy across time for each of the 36 bins
        energy_per_bin = np.mean(chromagram, axis=1)
        
        # The bin with the highest sustained energy is usually the Rukooz in Maqam music.
        rukooz_index = np.argmax(energy_per_bin)
        return rukooz_index