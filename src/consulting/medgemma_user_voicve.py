import streamlit as st
import requests
import json
from pathlib import Path
import tempfile
from faster_whisper import WhisperModel

# ---------------------------
# Configuration
# ---------------------------
PROMPT_FILE = "medgemma_prompt.md"
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "alibayram/medgemma:latest"

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="MedGemma Analyzer",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------
# Functions
# ---------------------------
@st.cache_resource
def load_whisper_model():
    """Load Whisper model once"""
    return WhisperModel("small", device="cpu")

def check_ollama(base_url=OLLAMA_BASE_URL):
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def generate_from_ollama(prompt, model_name=MODEL_NAME, base_url=OLLAMA_BASE_URL):
    """Send prompt to Ollama and get response"""
    url = f"{base_url}/api/generate"
    payload = {
        "model": model_name, 
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result.get("response")
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def extract_json_from_response(text):
    """Extract JSON from markdown code blocks if present"""
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

def transcribe_audio(audio_bytes):
    """Transcribe audio bytes to text"""
    model = load_whisper_model()
    
    # Save audio bytes to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        segments, info = model.transcribe(tmp_path, language="en")
        transcript = " ".join([segment.text for segment in segments])
        return transcript
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def format_analysis_summary(medgemma_json):
    """Format the analysis as a text summary"""
    summary = "=" * 70 + "\n"
    summary += "MEDGEMMA ANALYSIS SUMMARY\n"
    summary += "=" * 70 + "\n\n"
    
    if "condition" in medgemma_json:
        summary += f"Condition: {medgemma_json['condition']}\n"
        summary += f"Severity: {medgemma_json['severity']}\n"
        summary += f"\nRecommended Action:\n"
        summary += f"  {medgemma_json['recommended_action']}\n"
        summary += f"\nExplanation:\n"
        summary += f"  {medgemma_json['explanation']}\n"
        
        if "features" in medgemma_json:
            features = medgemma_json['features']
            summary += f"\nKey Features:\n"
            if features.get('colors'):
                summary += f"  - Colors: {', '.join(features['colors'])}\n"
            if features.get('body_locations'):
                summary += f"  - Location: {', '.join(features['body_locations'])}\n"
            if features.get('duration'):
                summary += f"  - Duration: {features['duration']}\n"
            if features.get('sensations'):
                summary += f"  - Sensations: {', '.join(features['sensations'])}\n"
    
    summary += "\n" + "=" * 70
    return summary

def analyze_transcript(transcript, prompt_template):
    """Analyze transcript with MedGemma"""
    prompt_filled = prompt_template.replace("{transcript}", transcript)
    medgemma_result = generate_from_ollama(prompt_filled)
    
    if medgemma_result is None:
        return None
    
    try:
        json_text = extract_json_from_response(medgemma_result)
        medgemma_json = json.loads(json_text)
        return medgemma_json
    except json.JSONDecodeError as e:
        st.error(f"Could not parse JSON response: {e}")
        return None

# ---------------------------
# Main App
# ---------------------------
def main():
    st.title("🩺 MedGemma Dermatology Analyzer")
    st.markdown("Upload a text file, record voice, or paste patient description")
    
    # Check Ollama status
    st.sidebar.header("System Status")
    if check_ollama():
        st.sidebar.success("✅ Ollama is running")
    else:
        st.sidebar.error("❌ Ollama is not running")
        st.error("Please start Ollama with: `ollama serve`")
        st.stop()
    
    st.sidebar.markdown(f"**Model:** {MODEL_NAME}")
    
    # Load prompt template
    if not Path(PROMPT_FILE).exists():
        st.error(f"❌ Prompt file not found: {PROMPT_FILE}")
        st.stop()
    
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "🎤 Voice Recording", "📁 Upload File"])
    
    transcript = None
    
    # Tab 1: Text Input
    with tab1:
        st.markdown("### Type or paste patient description")
        transcript_text = st.text_area(
            "Patient's description:",
            height=200,
            placeholder="Describe the skin condition, symptoms, duration, location, etc.",
            key="text_input"
        )
        if st.button("Analyze Text", type="primary", key="analyze_text"):
            if transcript_text.strip():
                transcript = transcript_text
            else:
                st.warning("⚠️ Please enter a description")
    
    # Tab 2: Voice Recording
    with tab2:
        st.markdown("### Record patient description using your microphone")
        st.info("🎙️ Use the audio recorder below to record the patient's description")
        
        # Streamlit's native audio input (requires Streamlit >= 1.28.0)
        audio_bytes = st.audio_input("Record audio", key="audio_recorder")
        
        if audio_bytes:
            st.audio(audio_bytes)
            
            if st.button("Transcribe & Analyze", type="primary", key="analyze_voice"):
                with st.spinner("Transcribing audio..."):
                    # Read the audio bytes
                    audio_data = audio_bytes.read()
                    transcript = transcribe_audio(audio_data)
                    
                    if transcript:
                        st.success("✅ Transcription complete!")
                        with st.expander("View transcript"):
                            st.write(transcript)
                    else:
                        st.error("❌ Transcription failed")
    
    # Tab 3: File Upload
    with tab3:
        st.markdown("### Upload a text or audio file")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Text File (.txt)**")
            text_file = st.file_uploader(
                "Upload text file",
                type=['txt'],
                help="Upload a .txt file with patient description",
                key="text_file_upload"
            )
            
            if text_file is not None:
                transcript_file = text_file.read().decode("utf-8")
                st.text_area("File content:", value=transcript_file, height=150, disabled=True)
                if st.button("Analyze Text File", type="primary", key="analyze_text_file"):
                    transcript = transcript_file
        
        with col2:
            st.markdown("**Audio File (.wav, .mp3, .m4a)**")
            audio_file = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="Upload an audio file to transcribe and analyze",
                key="audio_file_upload"
            )
            
            if audio_file is not None:
                st.audio(audio_file)
                if st.button("Transcribe & Analyze Audio", type="primary", key="analyze_audio_file"):
                    with st.spinner("Transcribing audio file..."):
                        audio_data = audio_file.read()
                        transcript = transcribe_audio(audio_data)
                        
                        if transcript:
                            st.success("✅ Transcription complete!")
                            with st.expander("View transcript"):
                                st.write(transcript)
    
    # Analyze if we have a transcript
    if transcript and transcript.strip():
        with st.spinner("Analyzing with MedGemma... Please wait."):
            analysis = analyze_transcript(transcript, prompt_template)
            
            if analysis:
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Format and display summary
                summary_text = format_analysis_summary(analysis)
                
                st.text_area(
                    "Summary:",
                    value=summary_text,
                    height=400,
                    disabled=True
                )
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Summary (.txt)",
                        data=summary_text,
                        file_name="medgemma_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download Full JSON",
                        data=json.dumps(analysis, indent=2),
                        file_name="medgemma_analysis.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.error("❌ Failed to get analysis from MedGemma")

if __name__ == "__main__":
    main()