import numpy as np
import os
import soundfile as sf

class SyntheticDataGenerator:
    """Generates synthetic audio files for Maqam training (36-bin resolution)."""
    
    def __init__(self, output_dir="data_synthetic_36", sr=22050):
        self.output_dir = output_dir
        self.sr = sr
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_tone(self, freq, duration):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        # Add some harmonics for realism (HPSS easier to test)
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
        signal += 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        signal += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        return signal

    def generate_maqam_samples(self, maqam_name, scale_notes, n_samples=50):
        """
        Generates random melodies based on a scale.
        scale_notes: List of frequencies (Hz)
        """
        maqam_dir = os.path.join(self.output_dir, maqam_name)
        if not os.path.exists(maqam_dir):
            os.makedirs(maqam_dir)
            
        print(f"Generating {n_samples} samples for {maqam_name}...")

        for i in range(n_samples):
            audio = np.array([])
            melody_indices = [0] # Start at root
            
            # Longer walks for better Seyir training
            length = np.random.randint(40, 80)
            
            for _ in range(length):
                last_idx = melody_indices[-1]
                # Weighted random walk: favor small steps
                step = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.3, 0.2, 0.3, 0.1])
                new_idx = np.clip(last_idx + step, 0, len(scale_notes) - 1)
                melody_indices.append(new_idx)
                
            melody_indices.append(0) # resolving to tonic
            
            for idx in melody_indices:
                freq = scale_notes[idx]
                dur = np.random.uniform(0.2, 0.6)
                tone = self.generate_tone(freq, dur)
                audio = np.concatenate((audio, tone))
                
            filename = os.path.join(maqam_dir, f"{maqam_name}_{i}.wav")
            sf.write(filename, audio, self.sr)

if __name__ == "__main__":
    gen = SyntheticDataGenerator(output_dir="data_synthetic_36")
    
    # Fundamental frequency for D (Re) = 293.66 Hz
    # 36-TET intervals from D
    def get_freq(semitones_from_D):
        # semitones can be fractional. 1 semitone = 3 steps in 36-TET (since 12 * 3 = 36)
        # Actually in 36-TET, 1 semitone is exactly 3 bins.
        # But we can just use 2^(n/12) logic directly or 2^(k/36).
        return 293.66 * (2 ** (semitones_from_D / 12.0))

    # Scales defined by intervals from Tonic (D) in semitones
    # Bayati: D, E-half-flat (-1.5), F (-0.5?), no wait.
    # D=0. E-half-flat is ~1.5 semitones above D? No.
    # Standard Bayati: D (0), E-half-flat (1.5), F (2.5), G (3.5), A (5.5)?
    # Let's map approximate semitones to Quarter tones.
    # E half flat is usually 3/4 tone = 1.5 semitones.
    
    # 1. Bayati (D, Eq, F, G, A, Bb, C, D)
    # Eq (E half flat) ~ 1.5 semitones = 150 cents
    # F = 3 semitones? No, D to F is minor third (3 semitones).
    bayati_intervals = [0, 1.5, 3.0, 5.0, 7.0, 8.0, 10.0, 12.0]
    bayati_scale = [get_freq(x) for x in bayati_intervals]
    gen.generate_maqam_samples("Bayati", bayati_scale, n_samples=100) # Increased for MLP
    
    # 2. Rast (C tonic for synthesis usually, but let's keep D for consistency to test TonicFinder)
    # Rast on D: D, E^ (1.75?), F#^ (3.75?)... 
    # Let's use standard C Rast intervals relative to C=261.63
    C = 261.63
    def get_freq_C(semitones): return C * (2**(semitones/12.0))
    # Rast: C, D, Ed, F, G, A, Bd, C
    # C(0), D(2), Eq(3.5), F(5), G(7), A(9), Bq(10.5), C(12)
    rast_intervals = [0, 2, 3.5, 5, 7, 9, 10.5, 12]
    rast_scale = [get_freq_C(x) for x in rast_intervals]
    gen.generate_maqam_samples("Rast", rast_scale, n_samples=100)
    
    # 3. Hijaz (D tonic)
    # D(0), Eb(1), F#(4), G(5), A(7), Bb(8), C(10), D(12)
    hijaz_intervals = [0, 1.0, 4.0, 5.0, 7.0, 8.0, 10.0, 12.0]
    hijaz_scale = [get_freq(x) for x in hijaz_intervals]
    gen.generate_maqam_samples("Hijaz", hijaz_scale, n_samples=100)
