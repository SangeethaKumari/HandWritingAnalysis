from cmd import PROMPT
from transformers import pipeline
from PIL import Image
import torch
from huggingface_hub import login
import json


login("""")

#IMAGE_PATH ="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/PXL_20251110_172428193.MP.jpg"
#IMAGE_PATH="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/skin_mole.jpg"
#IMAGE_PATH ="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/skin_allergy.jpeg"
IMAGE_PATH="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/blue_injury.jpeg"
#PROMPT_PATH ="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/medgemma_prompt.md"
PROMPT_PATH="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/simple_prompt.md"
TRANSCRIPT_PATH = "I dont know"
pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",  # example for multimodal
    torch_dtype=torch.bfloat16,
    device="cuda"
)
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

# Load your image and transcript
image = Image.open(IMAGE_PATH)   # ✅ not .read()
#transcript = open(TRANSCRIPT_PATH).read()
transcript ="I dont know"
system_prompt = open(PROMPT_PATH).read()  # your full medgemma prompt (with output format)
# 4. Combine text + image into structured prompt for MedGemma
full_prompt = system_prompt.replace("{transcript}", transcript)

conversation = [

    {
        "role": "user",
        "content": [
           # {"type": "text", "text": "Analyze this skin image and the transcript together."},
            {"type":"text","text":" You are a helpful clinical assistant. You triage dermatologic concerns given an image "
    "and symptom text. Provide two summaries: (1) a technical, doctor-facing summary with "
    "differential considerations and next-step suggestions; (2) a plain-language, patient-facing summary with safety guidance. "
    "You must include a disclaimer that this is not a diagnosis and is not medical advice."},
            {"type": "image", "image": image}
        ]
    }
]

# Single call — MedGemma will follow JSON instructions from the system prompt
medgemma_result = pipe(conversation, max_new_tokens=800)

# Access the JSON output
output_text = medgemma_result[0]["generated_text"]
#print("Raw output:\n", output_text)


# Extract assistant's response
assistant_response = None
for message in output_text:
    if message['role'] == 'assistant':
        assistant_response = message['content']
        break

# Depending on the structure, sometimes content is a string, sometimes a list
if isinstance(assistant_response, list):
    # Join all text parts
    output_text = ""
    for item in assistant_response:
        if item.get('type') == 'text':
            output_text += item['text'] + "\n"
else:
    output_text = assistant_response  # already a string

print("=== Assistant Analysis ===\n")
print(output_text)



#json_text = extract_json_from_response(output_text)
#medgemma_json = json.loads(json_text)
#summary_text = format_analysis_summary(medgemma_json)

#print(summary_text)




