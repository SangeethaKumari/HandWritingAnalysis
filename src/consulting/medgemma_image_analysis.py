from pathlib import Path

from PIL import Image
import torch
from transformers import pipeline


# ---------------------------
# Configuration
# ---------------------------
PROMPT_FILE = "medgemma_prompt.md"
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "alibayram/medgemma:latest"




IMAGE_PATH="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/blue_injury.jpeg"
PROMPT_PATH="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/simple_prompt.md"
TRANSCRIPT_PATH = "I dont know"


def build_conversation(prompt_text: str, image: Image.Image) -> list[dict]:
    """Prepare the conversation payload for the MedGemma pipeline."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": image},
            ],
        }
    ]






# Single call — MedGemma will follow JSON instructions from the system prompt
system_prompt =  """You are a helpful clinical assistant. You triage dermatologic concerns given an image "
    "and symptom text. Provide two summaries: (1) a technical, doctor-facing summary with "
    "differential considerations and next-step suggestions; (2) a plain-language, patient-facing summary with safety guidance. "
    "You must include a disclaimer that this is not a diagnosis and is not medical advice."# your full medgemma prompt (with output format)
"""
prompt_filled = system_prompt.replace("{transcript}", transcript)


medgemma_result = generate_from_ollama(system_prompt)
print("Raw output:\n", medgemma_result)







