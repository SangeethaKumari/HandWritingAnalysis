import streamlit as st
import speech_recognition as sr
import pyttsx3
import requests
import json

#sudo apt-get install python-pyaudio python3-pyaudio
#sudo apt-get install python3-pyaudio
#sudo apt-get install portaudio19-dev python3-dev

# --- Streamlit UI ---
st.title("🦙 Local Voice Chatbot (Ollama + LLaMA)")

# --- Settings ---
OLLAMA_MODEL = "llama3.1"  # or any other model you’ve pulled (e.g. mistral, phi3)
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Voice recognizer ---
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def query_ollama(prompt):
    """Send a query to your local Ollama model."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()
    return data.get("response", "⚠️ No response from model")

# --- Record and process voice ---
if st.button("🎤 Speak"):
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        st.success("Processing voice...")

    try:
        user_text = recognizer.recognize_google(audio)
        st.write(f"🗣️ **You said:** {user_text}")

        # --- Query Ollama ---
        reply = query_ollama(user_text)
        st.write(f"🤖 **LLaMA:** {reply}")

        # --- Speak response ---
        engine.say(reply)
        engine.runAndWait()

    except sr.UnknownValueError:
        st.error("❌ Sorry, I couldn’t understand that.")
    except Exception as e:
        st.error(f"Error: {e}")
