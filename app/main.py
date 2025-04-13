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
import tempfile
import os
import subprocess
import logging
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis Helpdesk")

# Configure paths and static files
base_dir = Path(__file__).parent  # Remove .parent to point to app directory
static_dir = base_dir / "static"  # Now points to app/static (adjust based on your actual structure)
templates_dir = base_dir / "templates"

# Mount static files directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Load models
try:
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")
    tone_model = joblib.load(models_dir / "tone_model.pkl")
    logger.info("Tone model and label encoder loaded successfully")
except Exception as e:
    logger.error(f"Error loading tone model or label encoder: {str(e)}")
    # Create mock models as fallback
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    tone_model = RandomForestClassifier(n_estimators=10)
    tone_model.feature_names_in_ = [f'feature_{i}' for i in range(15)]
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['NEUTRAL', 'POSITIVE', 'NEGATIVE'])

# NLP pipeline
try:
    text_sentiment_pipeline = pipeline("sentiment-analysis")
    logger.info("Text sentiment pipeline loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentiment pipeline: {str(e)}")
    # Fallback function
    def mock_sentiment_analysis(text):
        return [{"label": "NEUTRAL", "score": 0.95}]
    text_sentiment_pipeline = mock_sentiment_analysis

# Helper function to convert audio formats
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
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(webm_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

def transcribe_wav_data(wav_data: bytes) -> str:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.5
    
    try:
        audio = sr.AudioData(wav_data, sample_rate=16000, sample_width=2)
        transcribed_text = recognizer.recognize_google(audio, language="en-US")
        logger.info(f"Transcription successful: {transcribed_text[:50]}...")
        return transcribed_text
    except sr.UnknownValueError:
        logger.info("No speech detected in audio")
        return ""
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return ""

def extract_features_from_wav(wav_data: bytes) -> np.ndarray:
    try:
        audio_buffer = io.BytesIO(wav_data)
        y, sr = librosa.load(audio_buffer, sr=16000, duration=30)  # Limit to 30s audio
        
        if len(y) == 0:
            logger.warning("Empty audio data detected")
            return np.zeros((1, 15))  # Return empty features for bad audio

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        pitch = librosa.piptrack(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        
        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(pitch),
            np.mean(rms)
        ]).reshape(1, -1)
        
        logger.info(f"Features extracted successfully, shape: {features.shape}")
        return features
        
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
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
        logger.info(f"Text sentiment analysis result: {label}")
        return "NEUTRAL"
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
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
        tone_result = label_encoder.inverse_transform([np.argmax(tone_probs)])[0]
        logger.info(f"Tone sentiment prediction: {tone_result}")
        return tone_result
    except Exception as e:
        logger.error(f"Tone prediction error: {str(e)}")
        return "NEUTRAL"  # Fallback to neutral on errors

def fuse_sentiments(tone_label: str, text_label: str) -> str:
    # If tone and text sentiments agree, return that sentiment
    # Otherwise prioritize text sentiment as it's typically more reliable
    if tone_label == text_label:
        return tone_label
    return text_label

def generate_suggestions(sentiment: str, text: str) -> list:
    """Generate helpdesk suggestions based on sentiment analysis and transcript."""
    suggestions = []
    
    if not text or text.strip() == "":
        return ["No speech detected. Please try again."]
    
    # Basic suggestion templates
    if sentiment == "NEGATIVE":
        suggestions = [
            "Customer sounds upset. Consider escalating this case.",
            "Acknowledge their frustration and offer immediate assistance.",
            "Use empathetic language and active listening."
        ]
    elif sentiment == "POSITIVE":
        suggestions = [
            "Customer seems satisfied. Consider asking for feedback.",
            "Good opportunity to mention additional services or products.",
            "Thank them for their positive engagement."
        ]
    else:  # NEUTRAL
        suggestions = [
            "Maintain professional tone and clarity.",
            "Ask follow-up questions to better understand their needs.",
            "Provide clear step-by-step information."
        ]
    
    # Add content-specific suggestions based on keywords
    lower_text = text.lower()
    if "problem" in lower_text or "issue" in lower_text or "not working" in lower_text:
        suggestions.append("Customer is reporting a technical issue. Consider using troubleshooting scripts.")
    
    if "price" in lower_text or "cost" in lower_text or "expensive" in lower_text:
        suggestions.append("Price concerns detected. Review available discounts or payment plans.")
    
    if "wait" in lower_text or "long time" in lower_text:
        suggestions.append("Customer may be frustrated with wait times. Acknowledge and apologize for any delays.")
    
    return suggestions

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
        
        # Generate helpdesk suggestions
        suggestions = generate_suggestions(final_sentiment, transcribed_text)
        
        return {
            "transcribed_text": transcribed_text,
            "text_sentiment": text_sentiment,
            "tone_sentiment": tone_sentiment,
            "final_sentiment": final_sentiment,
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Error in analyze_audio endpoint: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": tone_model is not None and label_encoder is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)