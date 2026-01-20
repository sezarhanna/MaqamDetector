"""
Jins (Ajnas) Library - The Building Blocks of Maqamat.

Based on research from MaqamWorld.com (https://www.maqamworld.com)

A Jins (plural: Ajnas) is a tetrachord or pentachord that defines the melodic character
of a maqam. Each maqam is typically composed of two jins:
- Jins 1 (Lower): The foundation, determines the maqam's primary character
- Jins 2 (Upper): Built on top, adds color and defines the upper register

In 36-bin resolution (6 bins per semitone = quarter-tone accuracy):
- 1 semitone (half step) = 3 bins
- 1 whole step = 6 bins
- 3/4 step (neutral second) = 4.5 bins ≈ 4 or 5 bins
- 1.5 steps (augmented second) = 9 bins
"""

import numpy as np

# ============================================================================
# JINS DEFINITIONS (36-bin intervals from root = 0)
# Source: https://www.maqamworld.com/en/jins.php
# ============================================================================

JINS_LIBRARY = {
    # =========================================================================
    # Standard Jins (Root at bottom)
    # =========================================================================
    
    # Jins Bayati - 4 notes
    # Intervals: 3/4 + 3/4 + 1 (whole) = 2.5 whole steps
    # Notes on D: D, E♭½ (half-flat), F, G
    "Bayati": {
        "size": 4,
        "intervals_steps": [0.75, 0.75, 1],  # In whole steps
        "intervals_bins": [0, 5, 9, 15],      # Cumulative 36-bin positions
        "notes_on_D": ["D", "E♭½", "F", "G"],
        "ghammaz": 15,  # Highest note (G)
        "character": "neutral",
        "family": "Bayati"
    },
    
    # Jins Rast - 5 notes (pentachord)
    # Intervals: 1 + 3/4 + 3/4 + 1 = 3.5 whole steps
    # Notes on C: C, D, E♭½, F, G
    "Rast": {
        "size": 5,
        "intervals_steps": [1, 0.75, 0.75, 1],
        "intervals_bins": [0, 6, 10, 15, 21],
        "notes_on_C": ["C", "D", "E♭½", "F", "G"],
        "ghammaz": 21,  # G
        "character": "neutral",
        "family": "Rast"
    },
    
    # Jins Nahawand - 5 notes (pentachord)
    # Intervals: 1 + 1/2 + 1 + 1 = 3.5 whole steps
    # Notes on C: C, D, E♭, F, G
    "Nahawand": {
        "size": 5,
        "intervals_steps": [1, 0.5, 1, 1],
        "intervals_bins": [0, 6, 9, 15, 21],
        "notes_on_C": ["C", "D", "E♭", "F", "G"],
        "ghammaz": 21,  # G
        "character": "minor",
        "family": "Nahawand"
    },
    
    # Jins Hijaz - 4 notes
    # Intervals: 1/2 + 1.5 + 1/2 = 2.5 whole steps
    # Notes on D: D, E♭, F#, G
    "Hijaz": {
        "size": 4,
        "intervals_steps": [0.5, 1.5, 0.5],
        "intervals_bins": [0, 3, 12, 15],  # Augmented 2nd = 9 bins
        "notes_on_D": ["D", "E♭", "F#", "G"],
        "ghammaz": 15,  # G
        "character": "hijaz",
        "family": "Hijaz"
    },
    
    # Jins Kurd - 4 notes
    # Intervals: 1/2 + 1 + 1 = 2.5 whole steps
    # Notes on D: D, E♭, F, G
    "Kurd": {
        "size": 4,
        "intervals_steps": [0.5, 1, 1],
        "intervals_bins": [0, 3, 9, 15],
        "notes_on_D": ["D", "E♭", "F", "G"],
        "ghammaz": 15,  # G
        "character": "minor",
        "family": "Kurd"
    },
    
    # Jins Sikah - 3 notes (trichord) - starts on half-flat!
    # Intervals: 3/4 + 1 = 1.75 whole steps
    # Notes: E♭½, F, G
    "Sikah": {
        "size": 3,
        "intervals_steps": [0.75, 1],
        "intervals_bins": [0, 5, 11],
        "notes": ["E♭½", "F", "G"],
        "ghammaz": 11,  # G
        "character": "neutral",
        "family": "Sikah",
        "starts_on_neutral": True
    },
    
    # Jins Ajam - 5 notes (pentachord) - Major
    # Intervals: 1 + 1 + 1/2 + 1 = 3.5 whole steps
    # Notes on C: C, D, E, F, G (same as Major scale first 5 notes)
    "Ajam": {
        "size": 5,
        "intervals_steps": [1, 1, 0.5, 1],
        "intervals_bins": [0, 6, 12, 15, 21],
        "notes_on_C": ["C", "D", "E", "F", "G"],
        "ghammaz": 21,  # G
        "character": "major",
        "family": "Ajam"
    },
    
    # Jins Saba - Ambiguous size (overlapping with Jins Hijaz)
    # Intervals: 3/4 + 3/4 + 1/2 = 2 whole steps
    # Notes on D: D, E♭½, F, G♭ (diminished 4th!)
    "Saba": {
        "size": 4,
        "intervals_steps": [0.75, 0.75, 0.5],
        "intervals_bins": [0, 5, 9, 12],  # Note: G♭ is a diminished 4th
        "notes_on_D": ["D", "E♭½", "F", "G♭"],
        "ghammaz": 9,  # F (3rd degree, not 4th)
        "character": "saba",
        "family": "Saba",
        "unique_structure": True  # Ghammaz on 3rd degree
    },
    
    # Jins Nikriz - 5 notes (pentachord)
    # Intervals: 1 + 1/2 + 1.5 + 1/2 = 3.5 whole steps
    # Notes on C: C, D, E♭, F#, G
    "Nikriz": {
        "size": 5,
        "intervals_steps": [1, 0.5, 1.5, 0.5],
        "intervals_bins": [0, 6, 9, 18, 21],
        "notes_on_C": ["C", "D", "E♭", "F#", "G"],
        "ghammaz": 21,  # G
        "character": "hijaz",  # Contains augmented 2nd
        "family": "Nikriz"
    },
    
    # Jins Jiharkah - 5 notes (pentachord)
    # Intervals: 1 + 1 + 1/2 + 1 (like Ajam but on F, with lowered 3rd/4th)
    # Notes on F: F, G, A, B♭, C
    "Jiharkah": {
        "size": 5,
        "intervals_steps": [1, 1, 0.5, 1],
        "intervals_bins": [0, 6, 12, 15, 21],
        "notes_on_F": ["F", "G", "A", "B♭", "C"],
        "ghammaz": 21,  # C
        "character": "major",
        "family": "Jiharkah",
        "note": "3rd and 4th degrees traditionally played slightly lower"
    },
    
    # Jins Athar Kurd - 4 notes
    # Intervals: 1/2 + 1.5 + 1/2 (like Hijaz but starts like Kurd)
    # Notes on D: D, E♭, F#, G
    "Athar_Kurd": {
        "size": 4,
        "intervals_steps": [0.5, 1.5, 0.5],
        "intervals_bins": [0, 3, 12, 15],
        "notes_on_D": ["D", "E♭", "F#", "G"],
        "ghammaz": 15,
        "character": "hijaz",
        "family": "Kurd"
    },
    
    # Jins Lami - 4 notes
    # Intervals: 1/2 + 1 + 1/2
    # Notes on D: D, E♭, F, G♭
    "Lami": {
        "size": 4,
        "intervals_steps": [0.5, 1, 0.5],
        "intervals_bins": [0, 3, 9, 12],
        "notes_on_D": ["D", "E♭", "F", "G♭"],
        "ghammaz": 12,
        "character": "minor",
        "family": "Lami"
    },
    
    # Jins Mustaar - 4 notes
    # Intervals: 1 + 3/4 + 3/4
    # Similar to Rast but as a tetrachord
    "Mustaar": {
        "size": 4,
        "intervals_steps": [1, 0.75, 0.75],
        "intervals_bins": [0, 6, 10, 15],
        "notes_on_G": ["G", "A", "B♭½", "C"],
        "ghammaz": 15,
        "character": "neutral",
        "family": "Rast"
    },
    
    # =========================================================================
    # Upper Jins (Root at top - inverted direction)
    # =========================================================================
    
    # Jins Upper Rast - 4 notes (root is highest note)
    # Intervals descending from C: C, B♭½, A, G (or ascending: G, A, B♭½, C)
    # Intervals: 1 + 3/4 + 3/4
    "Upper_Rast": {
        "size": 4,
        "intervals_steps": [1, 0.75, 0.75],
        "intervals_bins": [0, 6, 10, 15],  # From G upward
        "notes_ascending": ["G", "A", "B♭½", "C"],
        "ghammaz": 0,  # G (lowest note is the ghammaz for upper jins)
        "character": "neutral",
        "family": "Rast",
        "is_upper_jins": True
    },
    
    # Jins Upper Ajam - 4 notes
    # Intervals: 1 + 1 + 1/2
    "Upper_Ajam": {
        "size": 4,
        "intervals_steps": [1, 1, 0.5],
        "intervals_bins": [0, 6, 12, 15],
        "notes_ascending": ["G", "A", "B", "C"],
        "ghammaz": 0,
        "character": "major",
        "family": "Ajam",
        "is_upper_jins": True
    },
}

# ============================================================================
# MAQAM STRUCTURES (Jins 1 + Jins 2 + Optional Modulation)
# Source: https://www.maqamworld.com/en/maqam.php
# ============================================================================

MAQAM_STRUCTURE = {
    # =========================================================================
    # BAYATI FAMILY
    # =========================================================================
    
    # Maqam Bayati: Bayati on D (jins1) + Nahawand on G (jins2)
    "Bayati": {
        "jins1": "Bayati",
        "jins1_root": 0,
        "jins2": "Nahawand",
        "jins2_root": 15,  # Perfect 4th (G)
        "modulation_jins": ["Rast", "Sikah"],  # Common modulations
        "scale_bins": [0, 5, 9, 15, 21, 24, 30, 36],  # D E♭½ F G A B♭ C D
        "family": "Bayati"
    },
    
    # Maqam Husayni: Bayati on D + Jins Bayati on A
    "Husayni": {
        "jins1": "Bayati",
        "jins1_root": 0,
        "jins2": "Bayati",
        "jins2_root": 21,  # Perfect 5th (A)
        "modulation_jins": ["Nahawand"],
        "family": "Bayati"
    },
    
    # Maqam Bayati Shuri: Bayati on D + Hijaz on G
    "Bayati_Shuri": {
        "jins1": "Bayati",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 15,
        "family": "Bayati"
    },
    
    # =========================================================================
    # RAST FAMILY
    # =========================================================================
    
    # Maqam Rast: Rast on C + Upper Rast on G
    "Rast": {
        "jins1": "Rast",
        "jins1_root": 0,
        "jins2": "Upper_Rast",
        "jins2_root": 21,  # Perfect 5th (G)
        "modulation_jins": ["Nahawand", "Bayati"],
        "scale_bins": [0, 6, 10, 15, 21, 27, 31, 36],  # C D E♭½ F G A B♭½ C
        "family": "Rast"
    },
    
    # Maqam Suznak: Rast on C + Hijaz on G
    "Suznak": {
        "jins1": "Rast",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 21,
        "modulation_jins": ["Nahawand"],
        "family": "Rast"
    },
    
    # Maqam Nairuz: Rast on C + Nahawand on G
    "Nairuz": {
        "jins1": "Rast",
        "jins1_root": 0,
        "jins2": "Nahawand",
        "jins2_root": 21,
        "family": "Rast"
    },
    
    # =========================================================================
    # NAHAWAND FAMILY
    # =========================================================================
    
    # Maqam Nahawand: Nahawand on C + Hijaz or Kurd on G
    "Nahawand": {
        "jins1": "Nahawand",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 21,  # 5th degree (G)
        "alt_jins2": "Kurd",  # Alternative for ascending
        "modulation_jins": ["Rast"],
        "scale_bins": [0, 6, 9, 15, 21, 24, 33, 36],
        "family": "Nahawand"
    },
    
    # Maqam Nahawand Murassaa: Nahawand on C + Nahawand on G
    "Nahawand_Murassaa": {
        "jins1": "Nahawand",
        "jins1_root": 0,
        "jins2": "Nahawand",
        "jins2_root": 21,
        "family": "Nahawand"
    },
    
    # =========================================================================
    # HIJAZ FAMILY
    # =========================================================================
    
    # Maqam Hijaz: Hijaz on D + Nahawand or Rast on G
    "Hijaz": {
        "jins1": "Hijaz",
        "jins1_root": 0,
        "jins2": "Nahawand",
        "jins2_root": 15,  # 4th degree (G)
        "alt_jins2": "Rast",
        "modulation_jins": ["Bayati"],
        "scale_bins": [0, 3, 12, 15, 21, 24, 30, 36],
        "family": "Hijaz"
    },
    
    # Maqam Hijazkar: Hijaz on C + Hijaz on G
    "Hijazkar": {
        "jins1": "Hijaz",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 21,
        "modulation_jins": ["Nahawand"],
        "family": "Hijaz"
    },
    
    # Maqam Shadd Araban: Hijaz on D + Upper Rast on A
    "Shadd_Araban": {
        "jins1": "Hijaz",
        "jins1_root": 0,
        "jins2": "Upper_Rast",
        "jins2_root": 21,
        "family": "Hijaz"
    },
    
    # Maqam Zanjaran: Hijaz on D + Sikah on F
    "Zanjaran": {
        "jins1": "Hijaz",
        "jins1_root": 0,
        "jins2": "Sikah",
        "jins2_root": 12,  # 3rd degree (F#)
        "family": "Hijaz"
    },
    
    # =========================================================================
    # KURD FAMILY
    # =========================================================================
    
    # Maqam Kurd: Kurd on D + Nahawand on G
    "Kurd": {
        "jins1": "Kurd",
        "jins1_root": 0,
        "jins2": "Nahawand",
        "jins2_root": 15,
        "modulation_jins": ["Hijaz"],
        "scale_bins": [0, 3, 9, 15, 21, 24, 30, 36],
        "family": "Kurd"
    },
    
    # =========================================================================
    # SIKAH FAMILY
    # =========================================================================
    
    # Maqam Sikah: Sikah on E♭½ + Hijaz on F or Rast on G
    "Sikah": {
        "jins1": "Sikah",
        "jins1_root": 0,
        "jins2": "Rast",
        "jins2_root": 11,  # 3rd degree
        "modulation_jins": ["Bayati", "Hijaz"],
        "family": "Sikah",
        "note": "Starts on quarter-flat E"
    },
    
    # Maqam Huzam: Sikah on E♭½ + Hijaz on B♭½
    "Huzam": {
        "jins1": "Sikah",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 15,  # On the 4th degree (relative)
        "family": "Sikah"
    },
    
    # Maqam Iraq: Sikah on E♭½ + Bayati on B♭½
    "Iraq": {
        "jins1": "Sikah",
        "jins1_root": 0,
        "jins2": "Bayati",
        "jins2_root": 15,
        "family": "Sikah"
    },
    
    # =========================================================================
    # AJAM FAMILY
    # =========================================================================
    
    # Maqam Ajam (Ajam Ushayran): Ajam on Bb + Upper Ajam on F
    "Ajam": {
        "jins1": "Ajam",
        "jins1_root": 0,
        "jins2": "Upper_Ajam",
        "jins2_root": 21,  # 5th degree
        "modulation_jins": ["Nahawand"],
        "scale_bins": [0, 6, 12, 15, 21, 27, 33, 36],  # Major scale
        "family": "Ajam"
    },
    
    # Maqam Jiharkah: Jiharkah on F + Rast on C
    "Jiharkah": {
        "jins1": "Jiharkah",
        "jins1_root": 0,
        "jins2": "Rast",
        "jins2_root": 21,
        "family": "Ajam"
    },
    
    # Maqam Shawq Afza: Ajam on C + Sikah on G
    "Shawq_Afza": {
        "jins1": "Ajam",
        "jins1_root": 0,
        "jins2": "Sikah",
        "jins2_root": 21,
        "family": "Ajam"
    },
    
    # =========================================================================
    # SABA FAMILY
    # =========================================================================
    
    # Maqam Saba: Saba on D + Hijaz on G♭ (overlapping!)
    "Saba": {
        "jins1": "Saba",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 12,  # Diminished 4th! (G♭)
        "modulation_jins": ["Bayati", "Kurd"],
        "scale_bins": [0, 5, 9, 12, 15, 18, 27, 36],
        "family": "Saba",
        "note": "Unique overlapping structure with diminished 4th"
    },
    
    # Maqam Saba Zamzam: Saba on D + Saba on G♭
    "Saba_Zamzam": {
        "jins1": "Saba",
        "jins1_root": 0,
        "jins2": "Saba",
        "jins2_root": 12,
        "family": "Saba"
    },
    
    # =========================================================================
    # NIKRIZ FAMILY
    # =========================================================================
    
    # Maqam Nikriz: Nikriz on C + Nahawand or Hijaz on G
    "Nikriz": {
        "jins1": "Nikriz",
        "jins1_root": 0,
        "jins2": "Nahawand",
        "jins2_root": 21,
        "alt_jins2": "Upper_Rast",
        "modulation_jins": ["Hijaz"],
        "scale_bins": [0, 6, 9, 18, 21, 27, 30, 36],
        "family": "Nikriz"
    },
    
    # Maqam Nawa Athar: Nikriz on C + Hijaz on G
    "Nawa_Athar": {
        "jins1": "Nikriz",
        "jins1_root": 0,
        "jins2": "Hijaz",
        "jins2_root": 21,
        "family": "Nikriz"
    },
    
    # Maqam Athar Kurd: Athar Kurd on D + Nikriz on G
    "Athar_Kurd": {
        "jins1": "Athar_Kurd",
        "jins1_root": 0,
        "jins2": "Nikriz",
        "jins2_root": 15,
        "family": "Nikriz"
    },
}


class JinsAnalyzer:
    """Analyzes pitch sequences to identify jins patterns based on MaqamWorld data."""
    
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
        if len(note_sequence) < 3:
            return {"jins": "Unknown", "confidence": 0.0}
        
        # Get unique notes in sequence (scale degrees used)
        unique_notes = sorted(set(note_sequence))
        
        best_match = None
        best_score = 0
        
        for jins_name, jins_data in self.jins_library.items():
            jins_intervals = jins_data.get("intervals_bins", [])
            if not jins_intervals:
                continue
            
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
        
        # First try exact matches
        for maqam_name, structure in self.maqam_structure.items():
            if structure["jins1"] == jins1_name:
                if structure["jins2"] == jins2_name:
                    confidence = (jins1_match["confidence"] + jins2_match["confidence"]) / 2
                    return {
                        "maqam": maqam_name,
                        "jins1": jins1_name,
                        "jins2": jins2_name,
                        "confidence": confidence,
                        "family": structure.get("family", "Unknown")
                    }
                # Check alternative jins2
                if structure.get("alt_jins2") == jins2_name:
                    confidence = (jins1_match["confidence"] + jins2_match["confidence"]) / 2 * 0.9
                    return {
                        "maqam": maqam_name,
                        "jins1": jins1_name,
                        "jins2": jins2_name,
                        "confidence": confidence,
                        "family": structure.get("family", "Unknown"),
                        "variant": True
                    }
        
        # No exact match, return best guess based on jins1 family
        family = self.jins_library.get(jins1_name, {}).get("family", "Unknown")
        return {
            "maqam": f"{jins1_name}-based",
            "jins1": jins1_name,
            "jins2": jins2_name,
            "confidence": jins1_match["confidence"] * 0.6,
            "family": family
        }
    
    def analyze_full_sequence(self, sequence):
        """
        Full analysis pipeline: segment and identify both jins.
        """
        # Try different jins2 roots common in maqam music
        # 12 = dim 4th (Saba), 15 = P4, 18 = tritone, 21 = P5
        jins2_roots = [12, 15, 18, 21]
        
        best_result = None
        best_confidence = 0
        
        for root in jins2_roots:
            jins1_seq, jins2_seq = self.segment_into_jins(sequence, root)
            
            if len(jins1_seq) < 3 or len(jins2_seq) < 2:
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
            "confidence": 0.0,
            "family": "Unknown"
        }
    
    def get_maqam_scale(self, maqam_name):
        """
        Get the full scale (in 36-bin indices) for a maqam.
        
        Returns:
            List of bin indices representing the scale, or None if not found
        """
        structure = self.maqam_structure.get(maqam_name)
        if structure and "scale_bins" in structure:
            return structure["scale_bins"]
        return None
    
    def get_jins_intervals(self, jins_name):
        """
        Get the interval pattern for a jins.
        
        Returns:
            Dict with intervals in steps and bins
        """
        jins = self.jins_library.get(jins_name)
        if jins:
            return {
                "steps": jins.get("intervals_steps", []),
                "bins": jins.get("intervals_bins", []),
                "size": jins.get("size", 0)
            }
        return None
