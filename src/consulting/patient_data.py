import librosa
import torch
import requests
import json
from pathlib import Path
from faster_whisper import WhisperModel

# ---------------------------
# Configuration
# ---------------------------
AUDIO_FILE = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/Rash_conversation.m4a"
PROMPT_FILE = "medgemma_prompt.md"
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "alibayram/medgemma:latest"

# ---------------------------
# Initialize Whisper model
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#model = WhisperModel("small", device="cpu") # need to change this later to device to use cuda
model = WhisperModel("large")

# ---------------------------
# Functions
# ---------------------------
def check_ollama(base_url=OLLAMA_BASE_URL):
    """Check if local Ollama API is running"""
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            print("Ollama is running (CPU mode)")
            return True
        else:
            print(f" Ollama API not reachable, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Ollama not running. Start with: ollama serve\nError: {e}")
        return False

def generate_from_ollama(prompt, model_name=MODEL_NAME, base_url=OLLAMA_BASE_URL):
    """Send a CoStar prompt to the local Ollama model and return the text output"""
    url = f"{base_url}/api/generate"
    payload = {"model": model_name, "prompt": prompt}
    
    #response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    response = requests.post(url, json=payload)
    raw_text = response.text.strip()
    print("raw_text ", raw_text)
       

    return raw_text  # Fallback: return whatever text is there
   

def load_audio(file_path, sr=16000):
    """Load audio file (wav, mp3, m4a supported)"""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def transcribe_audio(file_path, language="en"):
    """Transcribe audio to text using faster_whisper"""
    segments, info = model.transcribe(file_path, language=language)
    transcript = " ".join([segment.text for segment in segments])
    return transcript

# ---------------------------
# Main Workflow
# ---------------------------
if __name__ == "__main__":
    # Check audio file
    if not Path(AUDIO_FILE).exists():
        raise FileNotFoundError(f"{AUDIO_FILE} not found")

    # Load audio
    print("Loading audio...")
    waveform, sr = load_audio(AUDIO_FILE)
    print(f"Audio loaded. Sample rate: {sr}, Duration: {len(waveform)/sr:.2f}s")

    # Transcribe
    print("Transcribing audio...")
    transcript = transcribe_audio(AUDIO_FILE)
    print("Transcript obtained:")
    print(transcript)

    # Load prompt template
    if not Path(PROMPT_FILE).exists():
        raise FileNotFoundError(f"{PROMPT_FILE} not found")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Fill in transcript
    prompt_filled = prompt_template.replace("{transcript}", transcript)

   
    # Check Ollama
    if not check_ollama():
        exit(1)

    # Send to Ollama MedGemma
    print("Sending data to MedGemma for analysis...")
    medgemma_result = generate_from_ollama(prompt_filled)
    print("resultfrom MedGemma for analysis...")

   

    if isinstance(medgemma_result, dict):
        # Already a dict — either an error or proper structure
        print("MedGemma returned a dict:")
        print(json.dumps(medgemma_result, indent=2))
    else:
    # It's a string — maybe raw model output
        try:
            medgemma_json = json.loads(medgemma_result)
            print(json.dumps(medgemma_json, indent=2))
        except json.JSONDecodeError:
            print("Output is not valid JSON, printing raw text:")
        print(medgemma_result)
