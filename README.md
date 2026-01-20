# Maqam Detector

A full-stack application to detect Arabic/Turkish Maqams from audio recordings using microtonal analysis and Markov models.

## Structure

*   `src/`: Python source code for audio processing and analysis.
*   `flutter_app/`: Flutter source code for the frontend (Web/Mobile).
*   `data_synthetic/`: Generated training data.
*   `maqam_database.json`: Trained model file.

## Setup & Running

### 1. Backend (Python)

The backend handles the rigorous audio analysis.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install fastapi uvicorn python-multipart soundfile
    ```

2.  **Run the API Server**:
    ```bash
    uvicorn src.api:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

### 2. Frontend (Flutter)

1.  **Install Flutter SDK**: [https://docs.flutter.dev/get-started/install](https://docs.flutter.dev/get-started/install)
2.  **Run the App**:
    Navigate to the `flutter_app` folder and run:
    ```bash
    cd flutter_app
    flutter run -d chrome
    ```

## Features

*   **Synthetic Data Generation**: Creates theoretically perfect Maqam scales for robust initial training.
*   **Microtonal Analysis**: Uses CQT (Constant-Q Transform) to detect 24-quarter-tone intervals.
*   **Markov Classification**: Predicts the Maqam based on the melody path (Seyir).
