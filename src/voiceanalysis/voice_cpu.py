"""
CPU-ONLY Voice-to-Symptom Extraction System
Supports .wav, .m4a, .mp3 (via librosa)
"""

import os
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import re

import numpy as np
import librosa  # For audio loading

import requests
import torch

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SymptomExtraction:
    conversation_id: str
    colors: List[str]
    textures: List[str]
    patterns: List[str]
    size_descriptors: List[str]
    sensations: List[str]
    severity: str
    body_locations: List[str]
    duration: Optional[str]
    progression: Optional[str]
    triggers: List[str]
    aggravating_factors: List[str]
    relieving_factors: List[str]
    associated_symptoms: List[str]
    previous_treatments: List[str]
    confidence: str
    key_concerns: List[str]
    original_text_snippets: List[str]
    extracted_at: str
    processing_time_seconds: float
    gpu_used: bool

# ============================================================================
# CPU SPEECH-TO-TEXT
# ============================================================================

class CPUSpeechToText:
    """CPU Whisper transcription using faster-whisper"""
    
    def __init__(self):
        print("🎤 Loading Whisper model (CPU mode)...")
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            print("✅ Whisper model loaded on CPU")
        except ImportError:
            print("❌ faster-whisper not installed!")
            print("   Install with: pip install faster-whisper")
            exit(1)

    def transcribe(self, audio_path: str) -> str:
        print(f"🎯 Transcribing audio (CPU)...")
        start_time = time.time()
        segments, _ = self.model.transcribe(audio_path, beam_size=5)
        text = " ".join([s.text for s in segments])
        transcription_time = time.time() - start_time
        print(f"✅ Transcription completed in {transcription_time:.2f}s")
        return text

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    """Client for local Ollama API (CPU)"""
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print("✅ Ollama is running (CPU mode)")
            else:
                raise Exception("Ollama API not reachable")
        except:
            print("❌ Ollama not running. Start with: ollama serve")
            exit(1)

    def generate(self, prompt: str, model: str = "llama3.1", temperature: float = 0.0) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "options": {"num_predict": 2000, "num_gpu": 0}  # CPU mode
        }
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            return resp.json()['response']
        else:
            raise Exception(f"Ollama API error: {resp.status_code}")

# ============================================================================
# SYMPTOM EXTRACTOR
# ============================================================================

class SymptomExtractor:
    """Extract symptoms using Ollama API"""
    
    def __init__(self, model: str = "llama3.1"):
        self.ollama = OllamaClient()
        self.model = model

    def create_prompt(self, transcription: str) -> str:
        return f"""You are a medical AI assistant specialized in dermatology. Extract skin-related symptoms from this patient conversation.

PATIENT CONVERSATION:
{transcription}

Extract information in this EXACT JSON format (use empty lists [] if not mentioned):

{{
  "colors": [],
  "textures": [],
  "patterns": [],
  "size_descriptors": [],
  "sensations": [],
  "severity": "mild/moderate/severe",
  "body_locations": [],
  "duration": "",
  "progression": "",
  "triggers": [],
  "aggravating_factors": [],
  "relieving_factors": [],
  "associated_symptoms": [],
  "previous_treatments": [],
  "confidence": "high/medium/low",
  "key_concerns": [],
  "original_text_snippets": []
}}

RULES:
- Extract ONLY what is explicitly mentioned
- Return ONLY valid JSON
- Use empty lists [] or empty strings for missing info
JSON:"""

    def extract(self, transcription: str, conversation_id: str) -> SymptomExtraction:
        start_time = time.time()
        prompt = self.create_prompt(transcription)
        response = self.ollama.generate(prompt, model=self.model)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except:
                data = self._empty_data()
        else:
            data = self._empty_data()
        processing_time = time.time() - start_time
        extraction = SymptomExtraction(
            conversation_id=conversation_id,
            extracted_at=datetime.now().isoformat(),
            processing_time_seconds=round(processing_time, 2),
            gpu_used=False,
            **data
        )
        return extraction

    def _empty_data(self) -> Dict:
        return {
            "colors": [], "textures": [], "patterns": [], "size_descriptors": [],
            "sensations": [], "severity": "unknown", "body_locations": [],
            "duration": None, "progression": None, "triggers": [],
            "aggravating_factors": [], "relieving_factors": [],
            "associated_symptoms": [], "previous_treatments": [],
            "confidence": "low", "key_concerns": [], "original_text_snippets": []
        }

# ============================================================================
# PIPELINE
# ============================================================================

class CPUVoiceSymptomPipeline:
    """CPU-Only pipeline with full .m4a/.mp3/.wav support"""
    
    def __init__(self, output_dir: str = "patient_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "transcriptions").mkdir(exist_ok=True)
        (self.output_dir / "extractions").mkdir(exist_ok=True)
        self.transcriber = CPUSpeechToText()
        self.extractor = SymptomExtractor()
        print("✅ CPU-ONLY pipeline initialized")

    def process_voice_input(self, audio_file_path: str) -> Dict:
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        total_start = time.time()

        # --- Load audio via librosa ---
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        data, samplerate = librosa.load(audio_file_path, sr=None)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        duration_seconds = len(data) / samplerate
        print(f"✅ Loaded audio ({duration_seconds:.2f}s, {samplerate}Hz)")

        # Copy to output folder
        audio_output_path = self.output_dir / "audio" / f"{conversation_id}.wav"
        import soundfile as sf
        sf.write(audio_output_path, data, samplerate)
        print(f"💾 Audio saved to: {audio_output_path}")

        # --- Transcription ---
        transcription = self.transcriber.transcribe(audio_file_path)
        with open(self.output_dir / "transcriptions" / f"{conversation_id}.txt", 'w') as f:
            f.write(transcription)

        # --- Symptom extraction ---
        extraction = self.extractor.extract(transcription, conversation_id)
        extraction_path = self.output_dir / "extractions" / f"{conversation_id}.json"
        with open(extraction_path, 'w') as f:
            json.dump(asdict(extraction), f, indent=2)

        total_time = time.time() - total_start
        print(f"\n✅ Total processing time: {total_time:.2f}s")
        return {
            'conversation_id': conversation_id,
            'transcription': transcription,
            'extraction': asdict(extraction),
            'total_processing_time': round(total_time, 2)
        }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pipeline = CPUVoiceSymptomPipeline()

    # Example M4A file
    audio_file_path = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/Rash_conversation.m4a"
    result = pipeline.process_voice_input(audio_file_path)
    print(json.dumps(result, indent=2))
