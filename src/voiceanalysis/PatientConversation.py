"""
Complete Voice-to-Symptom Extraction System
Records patient voice → Transcribes → Extracts symptoms → Saves results

Installation:
pip install anthropic openai-whisper sounddevice scipy numpy pandas pyaudio
"""

import os
import json
import wave
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading

# Audio recording
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as write_wav

# Speech-to-text
import whisper

# LLM extraction
from anthropic import Anthropic


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PatientConversation:
    """Complete patient conversation record"""
    conversation_id: str
    timestamp: str
    audio_file: str
    transcription: str
    duration_seconds: float
    
    
@dataclass
class SymptomExtraction:
    """Extracted symptoms from conversation"""
    conversation_id: str
    
    # Visual characteristics
    colors: List[str]
    textures: List[str]
    patterns: List[str]
    size_descriptors: List[str]
    
    # Physical sensations
    sensations: List[str]
    severity: str
    
    # Location
    body_locations: List[str]
    
    # Temporal information
    duration: Optional[str]
    progression: Optional[str]
    
    # Context
    triggers: List[str]
    aggravating_factors: List[str]
    relieving_factors: List[str]
    
    # Additional
    associated_symptoms: List[str]
    previous_treatments: List[str]
    
    # Meta
    confidence: str
    key_concerns: List[str]
    original_text_snippets: List[str]
    
    # Processing metadata
    extracted_at: str
    processing_time_seconds: float


# ============================================================================
# VOICE RECORDING
# ============================================================================

class VoiceRecorder:
    """Records audio from microphone"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_data = []
        
    def record_until_silence(self, silence_duration: float = 2.0, 
                            silence_threshold: float = 0.01) -> np.ndarray:
        """
        Record audio until silence is detected.
        
        Args:
            silence_duration: Seconds of silence to stop recording
            silence_threshold: Amplitude threshold for silence detection
        """
        print("🎤 Recording started... (speak now, will auto-stop after silence)")
        
        self.audio_data = []
        self.is_recording = True
        silence_start = None
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            
            # Calculate RMS (volume level)
            volume = np.sqrt(np.mean(indata**2))
            
            if self.is_recording:
                self.audio_data.append(indata.copy())
                
                # Detect silence
                nonlocal silence_start
                if volume < silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        self.is_recording = False
                else:
                    silence_start = None
        
        # Start recording stream
        with sd.InputStream(callback=audio_callback, 
                           channels=1, 
                           samplerate=self.sample_rate):
            while self.is_recording:
                sd.sleep(100)
        
        print("✅ Recording stopped (silence detected)")
        
        # Combine all audio chunks
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            return audio_array
        return np.array([])
    
    def record_fixed_duration(self, duration: float) -> np.ndarray:
        """Record for a fixed duration in seconds"""
        print(f"🎤 Recording for {duration} seconds... (speak now)")
        
        audio_data = sd.rec(int(duration * self.sample_rate), 
                           samplerate=self.sample_rate, 
                           channels=1,
                           dtype='float32')
        sd.wait()  # Wait for recording to complete
        
        print("✅ Recording completed")
        return audio_data.flatten()
    
    def save_audio(self, audio_data: np.ndarray, filepath: str):
        """Save audio to WAV file"""
        # Convert to int16 for WAV format
        audio_int16 = (audio_data * 32767).astype(np.int16)
        write_wav(filepath, self.sample_rate, audio_int16)
        print(f"💾 Audio saved to: {filepath}")


# ============================================================================
# SPEECH-TO-TEXT
# ============================================================================

class SpeechToText:
    """Transcribe audio to text using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model.
        
        Args:
            model_size: tiny, base, small, medium, large
                       (larger = more accurate but slower)
        """
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print("✅ Whisper model loaded")
    
    def transcribe_audio_file(self, audio_path: str) -> Dict:
        """
        Transcribe audio file to text.
        
        Returns:
            Dict with 'text', 'segments', 'language'
        """
        print(f"🎯 Transcribing audio: {audio_path}")
        result = self.model.transcribe(audio_path)
        print("✅ Transcription completed")
        return result
    
    def transcribe_array(self, audio_data: np.ndarray) -> Dict:
        """Transcribe numpy audio array directly"""
        print("🎯 Transcribing audio...")
        result = self.model.transcribe(audio_data)
        print("✅ Transcription completed")
        return result


# ============================================================================
# SYMPTOM EXTRACTOR (LLM)
# ============================================================================

class SymptomExtractor:
    """Extract symptoms using Claude LLM"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def create_prompt(self, transcription: str) -> str:
        """Create extraction prompt"""
        return f"""You are a medical AI assistant specialized in dermatology. Extract skin-related symptoms from this patient conversation transcript.

PATIENT CONVERSATION TRANSCRIPT:
{transcription}

Extract the following information in JSON format:

{{
  "colors": ["color descriptors: red, pink, purple, etc."],
  "textures": ["rough, smooth, bumpy, scaly, flaky, etc."],
  "patterns": ["spotty, patchy, widespread, circular, etc."],
  "size_descriptors": ["small, large, spreading, coin-sized, etc."],
  "sensations": ["itchy, burning, painful, tender, tingling, etc."],
  "severity": "mild/moderate/severe",
  "body_locations": ["face, arms, legs, scalp, etc."],
  "duration": "how long (e.g., '3 days', '2 weeks', 'months')",
  "progression": "getting worse/better/stable/fluctuating",
  "triggers": ["sun, food, stress, products, etc."],
  "aggravating_factors": ["heat, scratching, sweating, etc."],
  "relieving_factors": ["cold, cream, medication, etc."],
  "associated_symptoms": ["fever, fatigue, swelling, etc."],
  "previous_treatments": ["creams, medications, home remedies mentioned"],
  "confidence": "high/medium/low",
  "key_concerns": ["patient's main worries"],
  "original_text_snippets": ["relevant direct quotes"]
}}

CRITICAL RULES:
- Extract ONLY what is explicitly stated or strongly implied
- Handle negations correctly ("not itchy" = exclude itchy)
- Use empty lists [] for missing information
- Pay attention to temporal context (was vs is vs will be)
- Capture exact patient language in snippets
- Be precise with body locations
- Consider severity from patient's tone and description

Return ONLY valid JSON, no additional text."""
    
    def extract(self, transcription: str, conversation_id: str) -> SymptomExtraction:
        """Extract symptoms from transcription"""
        print("🧠 Extracting symptoms with LLM...")
        start_time = time.time()
        
        prompt = self.create_prompt(transcription)
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        response_text = message.content[0].text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        data = json.loads(json_str)
        
        processing_time = time.time() - start_time
        
        # Create SymptomExtraction object
        extraction = SymptomExtraction(
            conversation_id=conversation_id,
            extracted_at=datetime.now().isoformat(),
            processing_time_seconds=round(processing_time, 2),
            **data
        )
        
        print(f"✅ Extraction completed in {processing_time:.2f}s")
        return extraction


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class VoiceSymptomPipeline:
    """Complete end-to-end pipeline"""
    
    def __init__(self, anthropic_api_key: str, 
                 output_dir: str = "patient_data",
                 whisper_model: str = "base"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "transcriptions").mkdir(exist_ok=True)
        (self.output_dir / "extractions").mkdir(exist_ok=True)
        
        # Initialize components
        self.recorder = VoiceRecorder()
        self.transcriber = SpeechToText(model_size=whisper_model)
        self.extractor = SymptomExtractor(api_key=anthropic_api_key)
        
        print("✅ Pipeline initialized")
    
    def process_voice_input(self, recording_mode: str = "auto") -> Dict:
        """
        Complete pipeline: Record → Transcribe → Extract
        
        Args:
            recording_mode: "auto" (stops on silence) or duration in seconds
        """
        # Generate unique ID
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"Starting new conversation: {conversation_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Record audio
        print("STEP 1: Recording Audio")
        print("-" * 40)
        
        if recording_mode == "auto":
            audio_data = self.recorder.record_until_silence()
        else:
            duration = float(recording_mode)
            audio_data = self.recorder.record_fixed_duration(duration)
        
        duration_seconds = len(audio_data) / self.recorder.sample_rate
        
        # Save audio
        audio_path = self.output_dir / "audio" / f"{conversation_id}.wav"
        self.recorder.save_audio(audio_data, str(audio_path))
        
        # Step 2: Transcribe
        print(f"\nSTEP 2: Speech-to-Text")
        print("-" * 40)
        
        transcription_result = self.transcriber.transcribe_array(audio_data)
        transcription_text = transcription_result['text']
        
        print(f"Transcription:\n\"{transcription_text}\"\n")
        
        # Save transcription
        transcription_path = self.output_dir / "transcriptions" / f"{conversation_id}.json"
        with open(transcription_path, 'w') as f:
            json.dump(transcription_result, f, indent=2)
        
        # Create conversation record
        conversation = PatientConversation(
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            audio_file=str(audio_path),
            transcription=transcription_text,
            duration_seconds=round(duration_seconds, 2)
        )
        
        # Step 3: Extract symptoms
        print(f"STEP 3: Extracting Symptoms")
        print("-" * 40)
        
        extraction = self.extractor.extract(transcription_text, conversation_id)
        
        # Save extraction
        extraction_path = self.output_dir / "extractions" / f"{conversation_id}.json"
        with open(extraction_path, 'w') as f:
            json.dump(asdict(extraction), f, indent=2)
        
        # Print summary
        self._print_summary(conversation, extraction)
        
        return {
            'conversation': asdict(conversation),
            'extraction': asdict(extraction)
        }
    
    def _print_summary(self, conversation: PatientConversation, 
                      extraction: SymptomExtraction):
        """Print extraction summary"""
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Conversation ID: {conversation.conversation_id}")
        print(f"Duration: {conversation.duration_seconds}s")
        print(f"Confidence: {extraction.confidence}")
        print(f"Severity: {extraction.severity}")
        
        if extraction.sensations:
            print(f"\n🔥 Sensations: {', '.join(extraction.sensations)}")
        
        if extraction.colors or extraction.textures:
            visual = []
            if extraction.colors:
                visual.extend(extraction.colors)
            if extraction.textures:
                visual.extend(extraction.textures)
            print(f"👁️  Visual: {', '.join(visual)}")
        
        if extraction.body_locations:
            print(f"📍 Locations: {', '.join(extraction.body_locations)}")
        
        if extraction.duration:
            print(f"⏱️  Duration: {extraction.duration}")
        
        if extraction.triggers:
            print(f"⚠️  Triggers: {', '.join(extraction.triggers)}")
        
        if extraction.key_concerns:
            print(f"💭 Key Concerns: {', '.join(extraction.key_concerns)}")
        
        print(f"\n{'='*60}\n")
    
    def batch_process_audio_files(self, audio_directory: str) -> List[Dict]:
        """Process multiple existing audio files"""
        results = []
        audio_files = list(Path(audio_directory).glob("*.wav"))
        
        print(f"Found {len(audio_files)} audio files to process\n")
        
        for idx, audio_file in enumerate(audio_files, 1):
            print(f"\nProcessing {idx}/{len(audio_files)}: {audio_file.name}")
            
            conversation_id = f"batch_{audio_file.stem}"
            
            # Transcribe
            transcription_result = self.transcriber.transcribe_audio_file(str(audio_file))
            transcription_text = transcription_result['text']
            
            # Extract
            extraction = self.extractor.extract(transcription_text, conversation_id)
            
            results.append({
                'file': str(audio_file),
                'transcription': transcription_text,
                'extraction': asdict(extraction)
            })
        
        return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    SETUP:
    1. pip install anthropic openai-whisper sounddevice scipy numpy
    2. Set ANTHROPIC_API_KEY environment variable
    3. Run this script
    """
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: Set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)
    
    # Initialize pipeline
    pipeline = VoiceSymptomPipeline(
        anthropic_api_key=api_key,
        whisper_model="base"  # Use "small" or "medium" for better accuracy
    )
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("VOICE SYMPTOM EXTRACTION SYSTEM")
        print("="*60)
        print("\n1. Record new patient conversation (auto-stop)")
        print("2. Record fixed duration (30 seconds)")
        print("3. Process existing audio files")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            result = pipeline.process_voice_input(recording_mode="auto")
            
        elif choice == "2":
            result = pipeline.process_voice_input(recording_mode="30")
            
        elif choice == "3":
            audio_dir = input("Enter audio directory path: ").strip()
            results = pipeline.batch_process_audio_files(audio_dir)
            print(f"\n✅ Processed {len(results)} files")
            
        elif choice == "4":
            print("\nGoodbye! 👋")
            break
        
        else:
            print("Invalid choice, try again")