"""
Jins (Ajnas) Library - The Building Blocks of Maqamat.

A Jins (plural: Ajnas) is a tetrachord or pentachord that defines the melodic character
of a maqam. Each maqam is typically composed of two jins:
- Jins 1 (Lower): The foundation, determines the maqam's primary character
- Jins 2 (Upper): Built on top, adds color and defines the upper register

In 36-bin resolution (6 bins per semitone = 1/6 tone accuracy):
- Quarter tone = 1.5 bins
- Semitone = 3 bins  
- Whole tone = 6 bins
- Neutral second (3/4 tone) ≈ 4-5 bins
"""

import numpy as np

# ============================================================================
# JINS DEFINITIONS (36-bin intervals from root = 0)
# ============================================================================

JINS_LIBRARY = {
    # --- Bayati Family (Neutral 2nd character) ---
    "Bayati": {
        "intervals": [0, 5, 9, 15],  # D - E♭ (neutral) - F - G
        "description": "Neutral 2nd, then whole, then whole",
        "character": "neutral"
    },
    
    # --- Rast Family (Neutral 3rd character) ---
    "Rast": {
        "intervals": [0, 6, 10, 15],  # C - D - E♭ (neutral) - F  
        "description": "Whole, then neutral 2nd, then whole",
        "character": "neutral"
    },
    
    # --- Nahawand/Minor Family ---
    "Nahawand": {
        "intervals": [0, 6, 9, 15],  # G - A - B♭ - C
        "description": "Whole, half, whole (minor tetrachord)",
        "character": "minor"
    },
    
    # --- Hijaz Family (Augmented 2nd character) ---
    "Hijaz": {
        "intervals": [0, 3, 12, 15],  # D - E♭ - F# - G
        "description": "Half, augmented 2nd, half",
        "character": "hijaz"
    },
    
    # --- Kurd Family ---
    "Kurd": {
        "intervals": [0, 3, 9, 15],  # D - E♭ - F - G
        "description": "Half, whole, whole (Phrygian)",
        "character": "minor"
    },
    
    # --- Sikah Family (Starts on neutral) ---
    "Sikah": {
        "intervals": [0, 4, 10, 15],  # E♭ (neutral) - F - G - A♭
        "description": "Neutral starting point",
        "character": "neutral"
    },
    
    # --- Ajam/Major Family ---
    "Ajam": {
        "intervals": [0, 6, 12, 15],  # B♭ - C - D - E♭
        "description": "Whole, whole, half (Major)",
        "character": "major"
    },
    
    # --- Saba Family ---
    "Saba": {
        "intervals": [0, 5, 9, 12],  # D - E♭ (neutral) - F - G♭
        "description": "Neutral 2nd, then whole, then diminished",
        "character": "saba"
    },
}

# ============================================================================
# MAQAM STRUCTURES (Jins 1 + Jins 2 combinations)
# ============================================================================

MAQAM_STRUCTURE = {
    # Maqam Bayati: Bayati on D (jins1) + Nahawand on G (jins2)
    "Bayati": {
        "jins1": "Bayati",
        "jins1_root": 0,      # Root at position 0 (relative)
        "jins2": "Nahawand", 
        "jins2_root": 15,     # 4th degree = 15 bins (perfect 4th)
        "typical_range": (0, 30),  # Roughly one octave + 4th
    },
    
    # Maqam Rast: Rast on C (jins1) + Rast on G (jins2)
    "Rast": {
        "jins1": "Rast",
        "jins1_root": 0,
        "jins2": "Rast",
        "jins2_root": 21,     # 5th degree = 21 bins (perfect 5th)
        "typical_range": (0, 36),
    },
    
    # Maqam Nahawand: Nahawand on C (jins1) + Hijaz on G (jins2)
    "Nahawand": {
        "jins1": "Nahawand",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 21,
        "typical_range": (0, 36),
    },
    
    # Maqam Hijaz: Hijaz on D (jins1) + Rast on G (jins2)  
    "Hijaz": {
        "jins1": "Hijaz",
        "jins1_root": 0,
        "jins2": "Rast",
        "jins2_root": 15,
        "typical_range": (0, 30),
    },
    
    # Maqam Kurd: Kurd on D (jins1) + Nahawand on G (jins2)
    "Kurd": {
        "jins1": "Kurd",
        "jins1_root": 0,
        "jins2": "Nahawand",
        "jins2_root": 15,
        "typical_range": (0, 30),
    },
    
    # Maqam Sikah: Sikah on E♭ (jins1) + Rast on A♭ (jins2)
    "Sikah": {
        "jins1": "Sikah",
        "jins1_root": 0,
        "jins2": "Rast",
        "jins2_root": 15,
        "typical_range": (0, 30),
    },
    
    # Maqam Ajam (Ajam Ushayran): Ajam on B♭ (jins1) + Ajam on F (jins2)
    "Ajam": {
        "jins1": "Ajam",
        "jins1_root": 0,
        "jins2": "Ajam",
        "jins2_root": 21,
        "typical_range": (0, 36),
    },
    
    # Maqam Saba: Saba on D (jins1) + Hijaz on G♭ (jins2)
    "Saba": {
        "jins1": "Saba",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 12,     # Diminished 4th
        "typical_range": (0, 27),
    },
    
    # Maqam Suznak: Rast on C (jins1) + Hijaz on G (jins2)
    "Suznak": {
        "jins1": "Rast",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 21,
        "typical_range": (0, 36),
    },
}


class JinsAnalyzer:
    """Analyzes pitch sequences to identify jins patterns."""
    
    def __init__(self, bins_per_octave=36):
        self.bins_per_octave = bins_per_octave
        self.jins_library = JINS_LIBRARY
        self.maqam_structure = MAQAM_STRUCTURE
    
    def match_jins(self, note_sequence, tolerance=2):
        """
        Match a note sequence to the most likely jins.
        
        Args:
            note_sequence: Array of bin indices (0-35) representing notes
            tolerance: How many bins of deviation allowed for matching
            
        Returns:
            dict with jins name and confidence score
        """
        if len(note_sequence) < 4:
            return {"jins": "Unknown", "confidence": 0.0}
        
        # Get unique notes in sequence (scale degrees used)
        unique_notes = sorted(set(note_sequence))
        
        best_match = None
        best_score = 0
        
        for jins_name, jins_data in self.jins_library.items():
            jins_intervals = jins_data["intervals"]
            
            # Check how many jins intervals are present in the sequence
            matches = 0
            for interval in jins_intervals:
                for note in unique_notes:
                    if abs(note - interval) <= tolerance:
                        matches += 1
                        break
            
            score = matches / len(jins_intervals)
            if score > best_score:
                best_score = score
                best_match = jins_name
        
        return {"jins": best_match or "Unknown", "confidence": best_score}
    
    def segment_into_jins(self, sequence, jins2_root=15):
        """
        Split a sequence into jins1 (lower) and jins2 (upper) regions.
        
        Args:
            sequence: Full normalized sequence (0 = tonic)
            jins2_root: Where jins2 starts (default: 15 = perfect 4th)
            
        Returns:
            Tuple of (jins1_sequence, jins2_sequence)
        """
        jins1_notes = []
        jins2_notes = []
        
        for note in sequence:
            if note < jins2_root:
                jins1_notes.append(note)
            else:
                # Normalize jins2 notes relative to jins2 root
                jins2_notes.append(note - jins2_root)
        
        return np.array(jins1_notes), np.array(jins2_notes)
    
    def predict_maqam_from_jins(self, jins1_match, jins2_match):
        """
        Given detected jins1 and jins2, predict the maqam.
        
        Args:
            jins1_match: Result from match_jins for lower tetrachord
            jins2_match: Result from match_jins for upper tetrachord
            
        Returns:
            Predicted maqam name and confidence
        """
        jins1_name = jins1_match["jins"]
        jins2_name = jins2_match["jins"]
        
        for maqam_name, structure in self.maqam_structure.items():
            if structure["jins1"] == jins1_name and structure["jins2"] == jins2_name:
                # Combined confidence
                confidence = (jins1_match["confidence"] + jins2_match["confidence"]) / 2
                return {
                    "maqam": maqam_name,
                    "jins1": jins1_name,
                    "jins2": jins2_name,
                    "confidence": confidence
                }
        
        # No exact match, return best guess based on jins1
        return {
            "maqam": f"{jins1_name}-based",
            "jins1": jins1_name,
            "jins2": jins2_name,
            "confidence": jins1_match["confidence"] * 0.7
        }
    
    def analyze_full_sequence(self, sequence):
        """
        Full analysis pipeline: segment and identify both jins.
        """
        # Try different jins2 roots common in maqam music
        jins2_roots = [12, 15, 18, 21]  # Diminished 4th, Perfect 4th, Tritone, Perfect 5th
        
        best_result = None
        best_confidence = 0
        
        for root in jins2_roots:
            jins1_seq, jins2_seq = self.segment_into_jins(sequence, root)
            
            if len(jins1_seq) < 4 or len(jins2_seq) < 2:
                continue
                
            jins1_match = self.match_jins(jins1_seq)
            jins2_match = self.match_jins(jins2_seq)
            
            result = self.predict_maqam_from_jins(jins1_match, jins2_match)
            
            if result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_result = result
                best_result["jins2_root"] = root
        
        return best_result or {
            "maqam": "Unknown",
            "jins1": "Unknown", 
            "jins2": "Unknown",
            "confidence": 0.0
        }
