import os
from SignalProcessor import SignalProcessor
from TonicFinder import TonicFinder
from SequenceNormalizer import SequenceNormalizer
from MaqamTrainer import MaqamTrainer

# Configuration
DATASET_PATH = "/Users/sezarhanna/Downloads/maqamidataset"
OUTPUT_DB = "maqam_database_hybrid.json"

# Mapping from folder names to Maqam names
MAPPING = {
    "Agm": "Ajam",
    "Byat": "Bayati",
    "Cord": "Kurd",
    "Hjaz": "Hijaz",
    "Nahawond": "Nahawand",
    "Rast": "Rast",
    "Sba": "Saba",
    "Sekah": "Sikah"
}

def main():
    print("Initializing components...")
    processor = SignalProcessor()
    finder = TonicFinder()
    normalizer = SequenceNormalizer()
    trainer = MaqamTrainer()
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} not found.")
        return

    print(f"Starting training on {DATASET_PATH}...")
    
    for folder_name, maqam_name in MAPPING.items():
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if os.path.exists(folder_path):
            print(f"-- Processing {folder_name} -> {maqam_name}")
            trainer.train_on_folder(maqam_name, folder_path, processor, finder, normalizer)
        else:
            print(f"Warning: Folder {folder_name} not found in dataset.")

    print("Finalizing and saving model...")
    trainer.finalize_and_save(markov_file=OUTPUT_DB)
    print("Training complete!")

if __name__ == "__main__":
    main()
