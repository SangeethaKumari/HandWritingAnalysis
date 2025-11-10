import streamlit as st
from streamlit_audio_recorder import st_audiorec
import speech_recognition as sr
import pyttsx3
import requests
import io

#uv add streamlit-audiorec
st.title("🦙 Local Voice Chatbot (Ollama + LLaMA)")

# --- Settings ---
OLLAMA_MODEL = "llama3.1"
OLLAMA_URL = "http://localhost:11434/api/generate"

engine = pyttsx3.init()

def query_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()
    return data.get("response", "⚠️ No response from model")

st.write("Click the record button below and speak:")

# --- Browser audio recorder ---
audio_bytes = st_audiorec()  # records audio from your Mac's microphone

if audio_bytes:
    st.success("Processing voice...")
    recognizer = sr.Recognizer()
    
    # Wrap bytes in a file-like object
    audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        user_text = recognizer.recognize_google(audio_data)
        st.write(f"🗣️ **You said:** {user_text}")

        reply = query_ollama(user_text)
        st.write(f"🤖 **LLaMA:** {reply}")

        engine.say(reply)
        engine.runAndWait()

    except sr.UnknownValueError:
        st.error("❌ Sorry, I couldn’t understand that.")
    except Exception as e:
        st.error(f"Error: {e}")
