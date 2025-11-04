"""
GPU-OPTIMIZED Voice-to-Symptom Extraction System
100% Local, Private, No API Costs - FAST on GPU!

GPU SETUP:
1. Install CUDA Toolkit (NVIDIA): https://developer.nvidia.com/cuda-downloads
2. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
3. Pull models:
   ollama pull llama3.1
4. Install Python packages:
   pip install sounddevice scipy numpy requests faster-whisper torch
5. GPU will be auto-detected and used!

## 🔍 Detailed Flow Diagram
```
User Selects Option 1 or 2
         ↓
pipeline.process_voice_input(mode)
         ↓
recorder.record_until_silence() OR recorder.record_fixed_duration()
         ↓
sd.InputStream() or sd.rec()  ← MICROPHONE ACCESSED HERE
         ↓
audio_callback(indata, ...)   ← AUDIO CHUNKS CAPTURED HERE
         ↓
self.audio_data.append(indata.copy())  ← STORED IN MEMORY
         ↓
Returns numpy array of audio data
         ↓
Saved to WAV file
         ↓
Sent to Whisper for transcription



"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import re

# Audio recording
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as write_wav

# Ollama API
import requests

# Check GPU availability
import torch


# ============================================================================
# GPU DETECTION
# ============================================================================

def check_gpu():
    """Check GPU availability and specs"""
    print("\n" + "="*60)
    print("GPU DETECTION")
    print("="*60)
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA Available: YES")
        print(f"✅ GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
        
        print("\n🚀 System will use GPU for acceleration!")
        return True
    else:
        print("⚠️  CUDA Not Available - Using CPU")
        print("   For GPU support, install CUDA Toolkit:")
        print("   https://developer.nvidia.com/cuda-downloads")
        return False


# ============================================================================
# DATA MODELS
# ============================================================================

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
    gpu_used: bool


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
        """Record audio until silence is detected"""
        print("🎤 Recording started... (speak now, will auto-stop after silence)")
        
        self.audio_data = []
        self.is_recording = True
        silence_start = None
        
        def audio_callback(indata, frames, time_info, status):
            volume = np.sqrt(np.mean(indata**2))
            
            if self.is_recording:
                self.audio_data.append(indata.copy())
                
                nonlocal silence_start
                if volume < silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        self.is_recording = False
                else:
                    silence_start = None
        
        with sd.InputStream(callback=audio_callback, 
                           channels=1, 
                           samplerate=self.sample_rate):
            while self.is_recording:
                sd.sleep(100)
        
        print("✅ Recording stopped (silence detected)")
        
        if self.audio_data:
            return np.concatenate(self.audio_data, axis=0)
        return np.array([])
    
    def record_fixed_duration(self, duration: float) -> np.ndarray:
        """Record for a fixed duration in seconds"""
        print(f"🎤 Recording for {duration} seconds... (speak now)")
        
        audio_data = sd.rec(int(duration * self.sample_rate), 
                           samplerate=self.sample_rate, 
                           channels=1,
                           dtype='float32')
        sd.wait()
        
        print("✅ Recording completed")
        return audio_data.flatten()
    
    def save_audio(self, audio_data: np.ndarray, filepath: str):
        """Save audio to WAV file"""
        audio_int16 = (audio_data * 32767).astype(np.int16)
        write_wav(filepath, self.sample_rate, audio_int16)
        print(f"💾 Audio saved to: {filepath}")


# ============================================================================
# GPU-ACCELERATED SPEECH-TO-TEXT
# ============================================================================

class GPUSpeechToText:
    """GPU-accelerated Whisper transcription"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        print(f"Loading Whisper model...")
        
        try:
            from faster_whisper import WhisperModel
            
            # Choose device and compute type based on GPU availability
            if self.use_gpu:
                device = "cuda"
                compute_type = "float16"  # GPU optimized
                print("🚀 Using GPU acceleration for Whisper")
            else:
                device = "cpu"
                compute_type = "int8"
                print("⚠️  Using CPU for Whisper")
            
            # Load model (base is good balance, use large for best accuracy)
            self.model = WhisperModel(
                "base",  # Options: tiny, base, small, medium, large
                device=device,
                compute_type=compute_type
            )
            
            print("✅ Whisper model loaded")
            
        except ImportError:
            print("❌ faster-whisper not installed!")
            print("   Install with: pip install faster-whisper")
            exit(1)
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file"""
        print(f"🎯 Transcribing audio (GPU: {self.use_gpu})...")
        start_time = time.time()
        
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        
        text = " ".join([segment.text for segment in segments])
        
        transcription_time = time.time() - start_time
        print(f"✅ Transcription completed in {transcription_time:.2f}s")
        
        return text


# ============================================================================
# OLLAMA CLIENT (GPU-ACCELERATED)
# ============================================================================

class OllamaClient:
    """Client for local Ollama API with GPU support"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.gpu_available = self.check_ollama_running()
    
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running and GPU status"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print("✅ Ollama is running")
                
                # Ollama automatically uses GPU if available
                if torch.cuda.is_available():
                    print("🚀 Ollama will use GPU acceleration")
                    return True
                else:
                    print("⚠️  Ollama running on CPU")
                    return False
        except:
            print("❌ Error: Ollama is not running!")
            print("   Start it with: ollama serve")
            exit(1)
    
    def generate(self, prompt: str, model: str = "llama3.1", 
                 temperature: float = 0.0) -> str:
        """Generate text using Ollama (GPU-accelerated)"""
        
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "options": {
                "num_predict": 2000,
                "num_gpu": 1 if self.gpu_available else 0  # Use GPU
            }
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Ollama API error: {response.status_code}")


# ============================================================================
# GPU-OPTIMIZED SYMPTOM EXTRACTOR
# ============================================================================

class GPUSymptomExtractor:
    """Extract symptoms using GPU-accelerated LLM"""
    
    def __init__(self, model: str = "llama3.1"):
        self.ollama = OllamaClient()
        self.model = model
        self.gpu_available = torch.cuda.is_available()
        
        print(f"✅ Using model: {model}")
        if self.gpu_available:
            print(f"🚀 GPU acceleration enabled")
    
    def create_prompt(self, transcription: str) -> str:
        """Create extraction prompt"""
        return f"""You are a medical AI assistant specialized in dermatology. Extract skin-related symptoms from this patient conversation.

PATIENT CONVERSATION:
{transcription}

Extract information in this EXACT JSON format (use empty lists [] if not mentioned):

{{
  "colors": ["red", "pink", etc."],
  "textures": ["rough", "smooth", "bumpy", etc."],
  "patterns": ["spotty", "patchy", etc."],
  "size_descriptors": ["small", "large", etc."],
  "sensations": ["itchy", "burning", etc."],
  "severity": "mild/moderate/severe",
  "body_locations": ["arms", "face", etc."],
  "duration": "3 days / 2 weeks / etc.",
  "progression": "getting worse/better/stable",
  "triggers": ["sun", "food", etc."],
  "aggravating_factors": ["heat", "scratching", etc."],
  "relieving_factors": ["cold", "cream", etc."],
  "associated_symptoms": ["fever", "fatigue", etc."],
  "previous_treatments": ["creams", "medications", etc."],
  "confidence": "high/medium/low",
  "key_concerns": ["spreading", "pain", etc."],
  "original_text_snippets": ["relevant quotes from patient"]
}}

RULES:
- Extract ONLY what is explicitly mentioned
- Handle negations: "not itchy" = exclude itchy
- Use empty lists [] for missing info
- Return ONLY valid JSON, no extra text
- Be precise and accurate

JSON:"""
    
    def extract(self, transcription: str, conversation_id: str) -> SymptomExtraction:
        """Extract symptoms from transcription (GPU-accelerated)"""
        print(f"🧠 Extracting symptoms with {self.model} (GPU: {self.gpu_available})...")
        start_time = time.time()
        
        prompt = self.create_prompt(transcription)
        
        # Get response from Ollama (GPU-accelerated)
        response = self.ollama.generate(prompt, model=self.model, temperature=0.0)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = json_str.replace('\n', ' ')
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                data = json.loads(json_str)
        else:
            print("⚠️  Warning: Could not extract JSON, using empty extraction")
            data = self._empty_data()
        
        processing_time = time.time() - start_time
        
        extraction = SymptomExtraction(
            conversation_id=conversation_id,
            extracted_at=datetime.now().isoformat(),
            processing_time_seconds=round(processing_time, 2),
            gpu_used=self.gpu_available,
            **data
        )
        
        print(f"✅ Extraction completed in {processing_time:.2f}s")
        if self.gpu_available:
            print(f"   ⚡ GPU-accelerated processing")
        
        return extraction
    
    def _empty_data(self) -> Dict:
        """Return empty extraction data"""
        return {
            "colors": [], "textures": [], "patterns": [], "size_descriptors": [],
            "sensations": [], "severity": "unknown", "body_locations": [],
            "duration": None, "progression": None, "triggers": [],
            "aggravating_factors": [], "relieving_factors": [],
            "associated_symptoms": [], "previous_treatments": [],
            "confidence": "low", "key_concerns": [], "original_text_snippets": []
        }


# ============================================================================
# GPU-OPTIMIZED PIPELINE
# ============================================================================

class GPUVoiceSymptomPipeline:
    """GPU-accelerated pipeline - FAST and FREE!"""
    
    def __init__(self, output_dir: str = "patient_data", 
                 llm_model: str = "llama3.1"):
        
        # Check GPU
        self.gpu_available = check_gpu()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "transcriptions").mkdir(exist_ok=True)
        (self.output_dir / "extractions").mkdir(exist_ok=True)
        
        self.recorder = VoiceRecorder()
        self.transcriber = GPUSpeechToText(use_gpu=self.gpu_available)
        self.extractor = GPUSymptomExtractor(model=llm_model)
        
        print("\n" + "="*60)
        print("✅ GPU-OPTIMIZED PIPELINE INITIALIZED")
        print("   • No API costs")
        print("   • 100% Private")
        print("   • GPU Acceleration: " + ("ON 🚀" if self.gpu_available else "OFF"))
        print("="*60 + "\n")
    
    def process_voice_input(self, recording_mode: str = "auto") -> Dict:
        """Complete GPU-accelerated pipeline: Record → Transcribe → Extract"""
        
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"NEW CONVERSATION: {conversation_id}")
        print(f"{'='*60}\n")
        
        total_start = time.time()
        
        # Step 1: Record audio
        print("STEP 1: Recording Audio")
        print("-" * 40)
        
        if recording_mode == "auto":
            audio_data = self.recorder.record_until_silence()
        else:
            duration = float(recording_mode)
            audio_data = self.recorder.record_fixed_duration(duration)
        
        duration_seconds = len(audio_data) / self.recorder.sample_rate
        
        audio_path = self.output_dir / "audio" / f"{conversation_id}.wav"
        self.recorder.save_audio(audio_data, str(audio_path))
        
        # Step 2: Transcribe (GPU-accelerated)
        print(f"\nSTEP 2: Speech-to-Text (GPU-Accelerated)")
        print("-" * 40)
        
        transcription = self.transcriber.transcribe(str(audio_path))
        
        print(f"\nTranscription:\n\"{transcription}\"\n")
        
        with open(self.output_dir / "transcriptions" / f"{conversation_id}.txt", 'w') as f:
            f.write(transcription)
        
        # Step 3: Extract symptoms (GPU-accelerated)
        print(f"STEP 3: Extracting Symptoms (GPU-Accelerated)")
        print("-" * 40)
        
        extraction = self.extractor.extract(transcription, conversation_id)
        
        extraction_path = self.output_dir / "extractions" / f"{conversation_id}.json"
        with open(extraction_path, 'w') as f:
            json.dump(asdict(extraction), f, indent=2)
        
        total_time = time.time() - total_start
        
        self._print_summary(transcription, extraction, duration_seconds, total_time)
        
        return {
            'conversation_id': conversation_id,
            'transcription': transcription,
            'extraction': asdict(extraction),
            'total_processing_time': round(total_time, 2)
        }
    
    def _print_summary(self, transcription: str, 
                      extraction: SymptomExtraction, 
                      duration: float, total_time: float):
        """Print extraction summary"""
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Audio Duration: {duration:.1f}s")
        print(f"Total Processing Time: {total_time:.1f}s")
        print(f"GPU Used: {'YES 🚀' if extraction.gpu_used else 'NO'}")
        print(f"Confidence: {extraction.confidence}")
        print(f"Severity: {extraction.severity}")
        
        if extraction.sensations:
            print(f"\n🔥 Sensations: {', '.join(extraction.sensations)}")
        
        if extraction.colors or extraction.textures:
            visual = extraction.colors + extraction.textures
            print(f"👁️  Visual: {', '.join(visual)}")
        
        if extraction.body_locations:
            print(f"📍 Locations: {', '.join(extraction.body_locations)}")
        
        if extraction.duration:
            print(f"⏱️  Duration: {extraction.duration}")
        
        if extraction.triggers:
            print(f"⚠️  Triggers: {', '.join(extraction.triggers)}")
        
        if extraction.key_concerns:
            print(f"💭 Concerns: {', '.join(extraction.key_concerns)}")
        
        print(f"\n{'='*60}\n")
        
        # Performance stats
        if extraction.gpu_used:
            print("⚡ GPU Acceleration: ACTIVE")
            print(f"   Processing Speed: {duration/total_time:.1f}x real-time")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """
    GPU SETUP INSTRUCTIONS:
    
    1. Install NVIDIA CUDA Toolkit:
       https://developer.nvidia.com/cuda-downloads
       
    2. Install Ollama:
       curl -fsSL https://ollama.com/install.sh | sh
    
    3. Pull models:
       ollama pull llama3.1          # 4.7GB
       # OR for better accuracy:
       ollama pull llama3.1:70b      # 40GB (needs 48GB+ GPU RAM)
    
    4. Install Python packages:
       pip install torch sounddevice scipy numpy requests faster-whisper
       # Make sure to install torch with CUDA support
    
    5. Run this script:
       python gpu_symptom_extractor.py
    """
    
    print("\n" + "="*60)
    print("GPU-ACCELERATED VOICE SYMPTOM EXTRACTION SYSTEM")
    print("="*60)
    
    # Initialize pipeline (will auto-detect GPU)
    pipeline = GPUVoiceSymptomPipeline(llm_model="llama3.1")
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("\n1. Record conversation (auto-stop on silence)")
        print("2. Record fixed duration (30 seconds)")
        print("3. Benchmark GPU performance")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            result = pipeline.process_voice_input(recording_mode="auto")
            print("\n✅ Data saved to patient_data/ directory")
            
        elif choice == "2":
            result = pipeline.process_voice_input(recording_mode="30")
            print("\n✅ Data saved to patient_data/ directory")
            
        elif choice == "3":
            print("\n🔥 Running GPU benchmark...")
            test_text = "I have red bumps on my arms that are itchy"
            start = time.time()
            _ = pipeline.extractor.extract(test_text, "benchmark")
            bench_time = time.time() - start
            print(f"✅ Benchmark complete: {bench_time:.2f}s")
            
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("Invalid choice")