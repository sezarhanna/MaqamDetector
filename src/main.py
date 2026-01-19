from SignalProcessor import SignalProcessor
from TonicFinder import TonicFinder
from SequenceNormalizer import SequenceNormalizer
from MarkovSeyirClassifier import MarkovSeyirClassifier
def run_maqam_engine(audio_path):
    # Initialize all steps
    step1 = SignalProcessor()
    step2 = TonicFinder()
    step3 = SequenceNormalizer()
    step4 = MarkovSeyirClassifier()

    # Execute Pipeline
    chroma = step1.get_chromagram(audio_path)
    rukooz = step2.find_rukooz(chroma)
    sequence = step3.normalize(chroma, rukooz)
    maqam, scores = step4.predict(sequence)

    print(f"Analysis Complete.")
    print(f"Detected Rukooz Bin: {rukooz}")
    print(f"Predicted Maqam: {maqam}")
    return maqam

# run_maqam_engine('my_oud_recording.wav')