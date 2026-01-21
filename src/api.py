from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import numpy as np
import json
import tempfile
from pathlib import Path

# Audio processing imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from .SignalProcessor import SignalProcessor
from .TonicFinder import TonicFinder
from .SequenceNormalizer import SequenceNormalizer
from .MaqamBrain import MaqamBrain
from .MaqamTrainer import MaqamTrainer
from .JinsLibrary import MAQAM_STRUCTURE

app = FastAPI(
    title="Maqam Detector API 2.0",
    description="AI-powered Arabic maqam detection with 36-bin microtonal resolution",
    version="2.0.0"
)

# CORS for Flutter Web / Any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for web UI
app.mount("/app", StaticFiles(directory="src/static", html=True), name="static")

# Initialize Components (36-bin)
processor = SignalProcessor(bins_per_octave=36)
finder = TonicFinder(bins_per_octave=36)
normalizer = SequenceNormalizer(bins_per_octave=36)
brain = MaqamBrain(bins_per_octave=36)

# Supported audio formats
SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}


def load_audio_file(file_path: str) -> tuple:
    """
    Load audio from any supported format using librosa.
    Returns (audio_array, sample_rate)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa not installed. Cannot process audio files.")
    
    # librosa handles MP3, WAV, FLAC, OGG, etc.
    audio, sr = librosa.load(file_path, sr=22050, mono=True)
    return audio, sr


@app.get("/")
def read_root():
    return {
        "status": "Maqam Detector 2.0 API is running",
        "version": "2.0.0",
        "endpoints": {
            "detect": "/predict (POST)",
            "contribute": "/contribute (POST)",
            "train": "/train (POST)",
            "stats": "/training-stats (GET)",
            "maqamat": "/maqamat (GET)"
        }
    }


@app.get("/maqamat")
def get_maqamat():
    """Get list of all supported maqamat with their jins structure."""
    return {
        "maqamat": list(MAQAM_STRUCTURE.keys()),
        "count": len(MAQAM_STRUCTURE),
        "structures": {
            name: {
                "jins1": struct["jins1"],
                "jins2": struct["jins2"],
                "family": struct.get("family", "Unknown")
            }
            for name, struct in MAQAM_STRUCTURE.items()
        }
    }


@app.get("/training-stats")
def get_training_stats():
    """Get statistics about available training data."""
    data_dir = Path("data")
    stats = {
        "files_per_maqam": {},
        "total_files": 0
    }
    
    if data_dir.exists():
        for maqam_folder in data_dir.iterdir():
            if maqam_folder.is_dir() and not maqam_folder.name.startswith('.'):
                audio_files = [
                    f for f in maqam_folder.iterdir() 
                    if f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                ]
                if audio_files:
                    stats["files_per_maqam"][maqam_folder.name] = len(audio_files)
                    stats["total_files"] += len(audio_files)
    
    return stats


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyze an audio file and predict its maqam.
    Supports: MP3, WAV, FLAC, M4A, OGG
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_AUDIO_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported format. Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"}
        )
    
    # Save to temp file (librosa needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name
    
    try:
        # Load and process audio
        audio, sr = load_audio_file(temp_path)
        
        # Get chromagram
        chroma = processor.get_chromagram(audio, sr=sr)
        
        # Find tonic
        rukooz = finder.find_rukooz(chroma)
        
        # Normalize sequence
        sequence = normalizer.normalize(chroma, rukooz)
        
        # Predict
        result = brain.predict(sequence)
        
        return {
            "filename": file.filename,
            "predicted_maqam": result["prediction"],
            "rukooz_bin": int(rukooz),
            "details": result
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/contribute")
async def contribute_training_data(
    maqam_name: str = Query(..., description="Name of the maqam"),
    file: UploadFile = File(...)
):
    """
    Upload a labeled audio file to contribute to training data.
    The file will be saved to data/{maqam_name}/ for future training.
    """
    # Validate maqam name
    valid_maqamat = set(MAQAM_STRUCTURE.keys())
    if maqam_name not in valid_maqamat:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unknown maqam: {maqam_name}",
                "valid_maqamat": list(valid_maqamat)
            }
        )
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_AUDIO_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported format. Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"}
        )
    
    # Create maqam folder if needed
    maqam_folder = Path("data") / maqam_name
    maqam_folder.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename to avoid overwrites
    import uuid
    safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = maqam_folder / safe_filename
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "message": f"File saved to training data for {maqam_name}",
            "filename": safe_filename,
            "maqam": maqam_name
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save file: {str(e)}"}
        )


@app.post("/train")
async def train(maqam_name: str = Query(None, description="Specific maqam to train, or leave empty for all")):
    """
    Trigger training on the available data.
    If maqam_name is provided, trains only that maqam.
    If not provided, trains on all available data.
    """
    trainer = MaqamTrainer(bins_per_octave=36)
    data_dir = Path("data")
    
    if not data_dir.exists():
        return JSONResponse(
            status_code=400,
            content={"error": "No data directory found. Upload some training data first."}
        )
    
    trained_maqamat = []
    
    if maqam_name:
        # Train specific maqam
        folder_path = data_dir / maqam_name
        if not folder_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"No training data found for {maqam_name}"}
            )
        trainer.train_on_folder(maqam_name, str(folder_path), processor, finder, normalizer)
        trained_maqamat.append(maqam_name)
    else:
        # Train all available maqamat
        for maqam_folder in data_dir.iterdir():
            if maqam_folder.is_dir() and not maqam_folder.name.startswith('.'):
                audio_files = [f for f in maqam_folder.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
                if audio_files:
                    trainer.train_on_folder(maqam_folder.name, str(maqam_folder), processor, finder, normalizer)
                    trained_maqamat.append(maqam_folder.name)
    
    if not trained_maqamat:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid audio files found for training."}
        )
    
    # Save models
    trainer.finalize_and_save()
    
    # Reload brain with new models
    brain._load_markov_models()
    brain._load_jins_models()
    
    return {
        "status": "success",
        "message": f"Training completed for {len(trained_maqamat)} maqamat",
        "trained_maqamat": trained_maqamat
    }


@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio analysis."""
    await websocket.accept()
    print("WebSocket Connected")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            if len(data) == 0:
                continue
            
            try:
                import io
                
                # Try to load as WAV first
                if SOUNDFILE_AVAILABLE:
                    with sf.SoundFile(io.BytesIO(data)) as f:
                        audio_chunk = f.read(dtype='float32')
                        sr = f.samplerate
                else:
                    # Fallback: assume raw float32 PCM
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    sr = 22050
                
                # Process
                chroma = processor.get_chromagram(audio_chunk, sr=sr)
                rukooz_idx = finder.find_rukooz(chroma)
                sequence = normalizer.normalize(chroma, rukooz_idx)
                prediction = brain.predict(sequence)
                
                response = {
                    "chromagram": chroma.tolist(),
                    "rukooz": int(rukooz_idx),
                    "prediction": prediction["prediction"],
                    "jins_analysis": prediction.get("jins_analysis", {}),
                    "confidence": prediction.get("confidence", 0)
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("WebSocket Disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
