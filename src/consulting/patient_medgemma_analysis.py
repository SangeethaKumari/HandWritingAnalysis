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
model = WhisperModel("small", device="cpu")

# ---------------------------
# Functions
# ---------------------------
def check_ollama(base_url=OLLAMA_BASE_URL):
    """Check if local Ollama API is running"""
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            print("✓ Ollama is running")
            return True
        else:
            print(f"✗ Ollama API not reachable, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Ollama not running. Start with: ollama serve\nError: {e}")
        return False

def generate_from_ollama(prompt, model_name=MODEL_NAME, base_url=OLLAMA_BASE_URL):
    """Send a prompt to Ollama and collect the full streaming response"""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model_name, 
        "prompt": prompt,
        "stream": False,  # Disable streaming for cleaner JSON responses
        "format": "json"  # Tell Ollama to output only JSON
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        print("\n[Response received from MedGemma]")
        
        # Extract the response text from non-streaming format
        if "response" in result:
            return result["response"]
        else:
            print(f"Unexpected response structure: {result}")
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error communicating with Ollama: {e}")
        return None

def load_audio(file_path, sr=16000):
    """Load audio file (wav, mp3, m4a supported)"""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def transcribe_audio(file_path, language="en"):
    """Transcribe audio to text using faster_whisper"""
    segments, info = model.transcribe(file_path, language=language)
    transcript = " ".join([segment.text for segment in segments])
    return transcript

def extract_json_from_response(text):
    """Extract JSON from markdown code blocks if present"""
    # Remove markdown code blocks if present
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    
    return text.strip()

# ---------------------------
# Main Workflow
# ---------------------------
if __name__ == "__main__":
    # Check audio file
    if not Path(AUDIO_FILE).exists():
        raise FileNotFoundError(f"✗ {AUDIO_FILE} not found")

    # Load audio
    print("Loading audio...")
    waveform, sr = load_audio(AUDIO_FILE)
    print(f"✓ Audio loaded. Sample rate: {sr}, Duration: {len(waveform)/sr:.2f}s")

    # Transcribe
    print("\nTranscribing audio...")
    transcript = transcribe_audio(AUDIO_FILE)
    print("✓ Transcript obtained:")
    print("-" * 50)
    print(transcript)
    print("-" * 50)

    # Load prompt template
    if not Path(PROMPT_FILE).exists():
        raise FileNotFoundError(f"✗ {PROMPT_FILE} not found")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Fill in transcript
    prompt_filled = prompt_template.replace("{transcript}", transcript)

    # Check Ollama
    print("\nChecking Ollama service...")
    if not check_ollama():
        exit(1)

    # Send to Ollama MedGemma
    print(f"\nSending to MedGemma ({MODEL_NAME}) for analysis...")
    medgemma_result = generate_from_ollama(prompt_filled)
    
    if medgemma_result is None:
        print("✗ Failed to get response from MedGemma")
        exit(1)
    
    print("\n" + "="*70)
    print("RAW MEDGEMMA RESPONSE:")
    print("="*70)
    print(medgemma_result)
    print("="*70)

    # Try to parse as JSON
    try:
        # Extract JSON from markdown if wrapped
        json_text = extract_json_from_response(medgemma_result)
        
        # Parse JSON
        medgemma_json = json.loads(json_text)
        
        print("\n" + "="*70)
        print("PARSED MEDGEMMA ANALYSIS:")
        print("="*70)
        print(json.dumps(medgemma_json, indent=2))
        
        # Display key information
        print("\n" + "="*70)
        print("SUMMARY:")
        print("="*70)
        if "condition" in medgemma_json:
            print(f"Condition: {medgemma_json['condition']}")
            print(f"Severity: {medgemma_json['severity']}")
            print(f"\nRecommended Action:")
            print(f"  {medgemma_json['recommended_action']}")
            print(f"\nExplanation:")
            print(f"  {medgemma_json['explanation']}")
            
            if "features" in medgemma_json:
                features = medgemma_json['features']
                print(f"\nKey Features:")
                if features.get('colors'):
                    print(f"  - Colors: {', '.join(features['colors'])}")
                if features.get('body_locations'):
                    print(f"  - Location: {', '.join(features['body_locations'])}")
                if features.get('duration'):
                    print(f"  - Duration: {features['duration']}")
                if features.get('sensations'):
                    print(f"  - Sensations: {', '.join(features['sensations'])}")
        print("="*70)
            
    except json.JSONDecodeError as e:
        print(f"\n✗ Response is not valid JSON: {e}")
        print("\nThe model did not follow the JSON format instruction.")
        print("This could mean:")
        print("  1. The model needs fine-tuning for structured output")
        print("  2. The prompt needs adjustment")
        print("  3. Try a different model (e.g., llama3 or mistral)")
        print("\nRaw response has been saved above for manual review.")