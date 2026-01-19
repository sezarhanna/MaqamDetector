import librosa
import numpy as np

class SignalProcessor:
    """Step 1: Convert raw audio into a high-resolution microtonal spectrogram."""
    def __init__(self, sr=22050, bins_per_octave=24):
        self.sr = sr
        self.bins_per_octave = bins_per_octave

    def get_chromagram(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        # CQT captures the 50-cent intervals (quarter-tones)
        cqt = np.abs(librosa.cqt(y, sr=self.sr, bins_per_octave=self.bins_per_octave))
        # Convert to Chroma to collapse octaves into pitch classes
        chroma = librosa.feature.chroma_cqt(C=cqt, bins_per_octave=self.bins_per_octave)
        return chroma