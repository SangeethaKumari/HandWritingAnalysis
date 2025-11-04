import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

print("🎤 Testing microphone...")

# Record 5 seconds
duration = 5
sample_rate = 16000

print(f"Recording for {duration} seconds... SPEAK NOW!")
audio = sd.rec(int(duration * sample_rate), 
               samplerate=sample_rate, 
               channels=1, 
               dtype='float32')
sd.wait()

print("✅ Recording complete!")

# Save to file
audio_int16 = (audio * 32767).astype(np.int16)
write('test_recording.wav', sample_rate, audio_int16)

print("💾 Saved to test_recording.wav")
print("Play it back to verify!")