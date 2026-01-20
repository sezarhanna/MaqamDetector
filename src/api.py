from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import numpy as np
import json
import soundfile as sf
from .SignalProcessor import SignalProcessor
from .TonicFinder import TonicFinder
from .SequenceNormalizer import SequenceNormalizer
from .MaqamBrain import MaqamBrain
from .MaqamTrainer import MaqamTrainer

app = FastAPI(title="Maqam Detector API 2.0")

from fastapi.staticfiles import StaticFiles

from fastapi.staticfiles import StaticFiles

# CORS for Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Web Visualizer
app.mount("/app", StaticFiles(directory="src/static", html=True), name="static")

# Serve Static Web Visualizer
app.mount("/app", StaticFiles(directory="src/static", html=True), name="static")

# Initialize Components (36-bin)
processor = SignalProcessor(bins_per_octave=36)
finder = TonicFinder(bins_per_octave=36)
normalizer = SequenceNormalizer(bins_per_octave=36)
brain = MaqamBrain(bins_per_octave=36)

@app.get("/")
def read_root():
    return {"status": "Maqam Detector 2.0 API is running"}

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket Connected")
    
    try:
        while True:
            # Receive audio chunk (bytes)
            # Flutter should send WAV-encoded bytes or raw PCM? 
            # For simplicity, let's assume Flutter sends a small WAV file chunk or we handle raw PCM.
            # Ideally: Raw Float32 PCM.
            data = await websocket.receive_bytes()
            
            # TODO: Convert bytes to numpy array
            # This is complex without a fixed header.
            # Strategy: For prototype, send small WAV files periodically via HTTP might be safer?
            # Or use a library like `soundfile` with `io.BytesIO`.
            
            # Assuming raw float32 for now (common in WebAudio)
            try:
                # audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Mock analysis for initial connection test if empty
                if len(data) == 0: continue

                # In a real stream, we'd buffer this. 
                # For this demo, we might expect a full "window" sent every 100ms.
                
                # Let's pivot: For the demo, receiving valid WAV bytes is easier.
                import io
                with sf.SoundFile(io.BytesIO(data)) as f:
                    audio_chunk = f.read(dtype='float32')
                    
                # 1. Signal Processing
                chroma = processor.get_chromagram(audio_chunk)
                
                # 2. Logic
                rukooz_idx = finder.find_rukooz(chroma)
                sequence = normalizer.normalize(chroma, rukooz_idx)
                prediction = brain.predict(sequence)
                
                response = {
                    "chromagram": chroma.tolist(), # 36 x Time
                    "rukooz": int(rukooz_idx),
                    "prediction": prediction["prediction"],
                    "confidence": prediction.get("mlp_scores", prediction.get("markov_scores", {})),
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("WebSocket Disconnected")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Legacy Endpoint: Full file upload"""
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chroma = processor.get_chromagram(temp_file)
        rukooz = finder.find_rukooz(chroma)
        sequence = normalizer.normalize(chroma, rukooz)
        result = brain.predict(sequence)
        
        return {
            "filename": file.filename,
            "predicted_maqam": result["prediction"],
            "rukooz_bin": int(rukooz),
            "details": result
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/train")
async def train(maqam_name: str, background_tasks: bool = True):
    """
    Trigger training for a specific Maqam folder in `data/`.
    Now expects `data/{maqam_name}` to exist with Real wav files.
    """
    trainer = MaqamTrainer(bins_per_octave=36)
    folder_path = os.path.join("data", maqam_name)
    
    if not os.path.exists(folder_path):
        return {"error": f"Folder data/{maqam_name} not found. Please download the dataset."}
        
    # We run synchronously for safety in this demo, or use BackgroundTasks
    trainer.train_on_folder(maqam_name, folder_path, processor, finder, normalizer)
    trainer.finalize_and_save()
    
    # Reload brain
    brain._load_markov_models()
    brain.mlp._load_model()
    
    return {"status": f"Training completed for {maqam_name}"}
