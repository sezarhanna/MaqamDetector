import numpy as np

class SequenceNormalizer:
    """Step 3: Normalize the pitch sequence relative to the detected Rukooz."""
    def __init__(self, bins_per_octave=36):
        self.bins_per_octave = bins_per_octave

    def normalize(self, chromagram, rukooz_index):
        # Get the most dominant note at every time step (The 'Melody Line')
        raw_sequence = np.argmax(chromagram, axis=0)
        
        # Shift every note so that Rukooz = 0 (Transposition)
        # This makes the model key-invariant.
        normalized_sequence = (raw_sequence - rukooz_index) % self.bins_per_octave
        return normalized_sequence