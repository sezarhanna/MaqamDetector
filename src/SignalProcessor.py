import librosa
import numpy as np

class SignalProcessor:
    """
    Step 1: Advanced Audio Preprocessing.
    - Harmonic-Percussive Source Separation (HPSS) to isolate melody.
    - Constant-Q Transform (CQT) with 36 bins per octave (1/6th tone).
    """
    def __init__(self, sr=22050, bins_per_octave=36):
        self.sr = sr
        self.bins_per_octave = bins_per_octave
        # We want meaningful pitch coverage. 7 octaves is standard for CQT.
        self.n_bins = bins_per_octave * 7 

    def get_chromagram(self, audio_path_or_y):
        """
        Computes the Chromagram from an audio file path or a loaded numpy array.
        """
        if isinstance(audio_path_or_y, str):
            y, sr = librosa.load(audio_path_or_y, sr=self.sr)
        else:
            y = audio_path_or_y
            
        # 1. HPSS: Isolate Harmonic component (melody) from Percussive (rhythm/noise)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # 2. CQT: High-resolution pitch analysis on the harmonic component
        cqt = np.abs(librosa.cqt(y_harmonic, sr=self.sr, 
                                 bins_per_octave=self.bins_per_octave, 
                                 n_bins=self.n_bins))
                                 
        # 3. Chroma: Fold into a single octave (Pitch Class Profile)
        chroma = librosa.feature.chroma_cqt(C=cqt, 
                                            bins_per_octave=self.bins_per_octave,
                                            n_octaves=7)
                                            
        return chroma