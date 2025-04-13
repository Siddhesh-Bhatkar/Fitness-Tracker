from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import io
import librosa
import joblib
import speech_recognition as sr
from pathlib import Path
from transformers import pipeline
import tempfile
import os
import subprocess

app = FastAPI()

# Configure paths and static files
base_dir = Path(__file__).parent.parent
static_dir = base_dir / "static"
templates_dir = base_dir / "templates"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Mock models for demo purposes
# In production, you would load actual trained models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
tone_model = RandomForestClassifier(n_estimators=10)
tone_model.feature_names_in_ = [f'feature_{i}' for i in range(15)]
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['NEUTRAL', 'POSITIVE', 'NEGATIVE'])

# NLP pipeline
try:
    text_sentiment_pipeline = pipeline("sentiment-analysis")
    print("Text sentiment pipeline loaded")
except Exception as e:
    print(f"Error loading sentiment pipeline: {str(e)}")
    # Fallback function
    def mock_sentiment_analysis(text):
        return [{"label": "NEUTRAL", "score": 0.95}]
    text_sentiment_pipeline = mock_sentiment_analysis

def convert_audio_to_wav(audio_data: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
        webm_file.write(audio_data)
        webm_path = webm_file.name
    
    wav_path = webm_path + ".wav"
    
    try:
        # Convert WebM/Opus to WAV using FFmpeg
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', webm_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            wav_path
        ]
        
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        with open(wav_path, 'rb') as wav_file:
            return wav_file.read()
            
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.decode())
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(webm_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass

def transcribe_wav_data(wav_data: bytes) -> str:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.5
    
    try:
        audio = sr.AudioData(wav_data, sample_rate=16000, sample_width=2)
        return recognizer.recognize_google(audio, language="en-US")
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

def extract_features_from_wav(wav_data: bytes) -> np.ndarray:
    try:
        audio_buffer = io.BytesIO(wav_data)
        y, sr = librosa.load(audio_buffer, sr=16000, duration=30)  # Limit to 30s audio
        
        if len(y) == 0:
            return np.zeros((1, 15))  # Return empty features for bad audio

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        pitch = librosa.piptrack(y=y, sr=sr)
        return np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(pitch),
            np.mean(librosa.feature.rms(y=y))
        ]).reshape(1, -1)
        
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return np.zeros((1, 15))  # Fallback empty features

def analyze_sentiment(text: str) -> str:
    if not text or text.strip() == "":
        return "NEUTRAL"
    
    try:
        result = text_sentiment_pipeline(text)[0]
        # Map HuggingFace sentiment labels to our format
        label = result['label']
        if isinstance(label, str):
            if "POSITIVE" in label or "positive" in label.lower():
                return "POSITIVE"
            elif "NEGATIVE" in label or "negative" in label.lower():
                return "NEGATIVE"
        return "NEUTRAL"
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return "NEUTRAL"

def predict_tone(wav_data: bytes) -> str:
    try:
        features = extract_features_from_wav(wav_data)
        
        # Ensure features match the expected shape
        if features.shape[1] != len(tone_model.feature_names_in_):
            # Adjust feature size if needed
            pad_width = ((0, 0), (0, len(tone_model.feature_names_in_) - features.shape[1]))
            features = np.pad(features, pad_width, mode='constant')
        
        tone_probs = tone_model.predict_proba(features)[0]
        return label_encoder.inverse_transform([np.argmax(tone_probs)])[0]
    except Exception as e:
        print(f"Tone prediction error: {str(e)}")
        return "NEUTRAL"  # Fallback to neutral on errors

def fuse_sentiments(tone_label: str, text_label: str) -> str:
    # If tone and text sentiments agree, return that sentiment
    # Otherwise prioritize text sentiment as it's typically more reliable
    if tone_label == text_label:
        return tone_label
    return text_label

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    audio_data = await file.read()
    
    try:
        # Convert audio to WAV format
        wav_data = convert_audio_to_wav(audio_data)
        
        # Transcribe audio
        transcribed_text = transcribe_wav_data(wav_data)
        
        # Analyze text sentiment
        text_sentiment = analyze_sentiment(transcribed_text) if transcribed_text.strip() else "NEUTRAL"
        
        # Analyze tone sentiment
        tone_sentiment = predict_tone(wav_data)
        
        # Fuse sentiments
        final_sentiment = fuse_sentiments(tone_sentiment, text_sentiment)
        
        return {
            "transcribed_text": transcribed_text,
            "text_sentiment": text_sentiment,
            "tone_sentiment": tone_sentiment,
            "final_sentiment": final_sentiment
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "ok"}