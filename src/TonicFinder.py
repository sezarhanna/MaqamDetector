import numpy as np

class TonicFinder:
    """Step 2: Analyze energy distribution to find the Rukooz (Tonic)."""
    def __init__(self, bins_per_octave=24):
        self.bins_per_octave = bins_per_octave

    def find_rukooz(self, chromagram):
        # Sum energy across time for each of the 24 bins
        energy_per_bin = np.mean(chromagram, axis=1)
        # The bin with the highest sustained energy is our Rukooz
        rukooz_index = np.argmax(energy_per_bin)
        return rukooz_index