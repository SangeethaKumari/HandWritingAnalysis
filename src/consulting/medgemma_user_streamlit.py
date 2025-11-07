import streamlit as st
import requests
import json
from pathlib import Path

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
    st.markdown("Upload a text file with patient description or paste the transcript directly")
    
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
    
    # Input section
    st.header("Input Patient Transcript")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload text file with patient description",
            type=['txt'],
            help="Upload a .txt file containing the patient's description"
        )
    
    with col2:
        # Or paste directly
        st.markdown("**Or paste transcript below:**")
    
    # Text input area
    if uploaded_file is not None:
        transcript = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded content:", value=transcript, height=200, disabled=True)
    else:
        transcript = st.text_area(
            "Patient's description:",
            height=200,
            placeholder="Paste or type the patient's description of symptoms here..."
        )
    
    # Analyze button
    if st.button("🔍 Analyze with MedGemma", type="primary", use_container_width=True):
        if not transcript.strip():
            st.warning("⚠️ Please provide a transcript to analyze")
        else:
            with st.spinner("Analyzing with MedGemma... Please wait."):
                analysis = analyze_transcript(transcript, prompt_template)
                
                if analysis:
                    # Format summary
                    summary_text = format_analysis_summary(analysis)
                    
                    # Display in text area
                    st.header("Analysis Results")
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