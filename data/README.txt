REQUIRED DATASET: Maqam-478
===========================

To train the Maqam Detector with real accuracy, we need real audio recordings.

1. DOWNLOAD
   - Go to: https://figshare.com/articles/dataset/Maqam-478_Dataset/13536854
   - Download the zip file (approx 3GB).

2. EXTRACT & ORGANIZE
   - Unzip the downloaded file.
   - You will find folders for different Maqams (e.g., 'Bayati', 'Rast', etc.).
   - Drag and drop the WAV files from each downloaded folder into the corresponding folder here in 'data/'.

   Example Structure:
   MaqamDetector/
   ├── data/
   │   ├── Bayati/
   │   │   ├── recitation_1.wav
   │   │   ├── recitation_2.wav
   │   ├── Rast/
   │   │   ├── recitation_3.wav
   │   │   └── ...
   │   ├── Suznak/
   │   │   ├── recitation_4.wav

3. TRAIN
   - Once files are placed, run the training command via the API or terminal.
   - Example (Terminal):
     curl -X POST "http://127.0.0.1:8000/train?maqam_name=Bayati"
