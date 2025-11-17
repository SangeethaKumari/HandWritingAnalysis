import ollama
from PIL import Image
import base64
from io import BytesIO

# --- Load inputs ---
#MODEL_NAME = "alibayram/medgemma:latest"
MODEL_NAME = "amsaravi/medgemma-4b-it:q6"

IMAGE_PATH="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/blue_tongue.jpeg"
PROMPT_PATH="/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/simple_prompt.md"
TRANSCRIPT_PATH = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/consulting/transcript.txt"
#transcript content , make sure the image and transcript are about the same thing.
# I have red itchy patches with small blisters on my arm for two days. It started after having some medicine. Scratching makes it worse. The affected area is about the size of my palm and hasn't spread much since yesterday.

with open(PROMPT_PATH, "r") as f:
    prompt = f.read()

with open(TRANSCRIPT_PATH, "r") as f:
    transcript = f.read()

# --- Encode the image to base64 ---
def encode_image_to_base64(IMAGE_PATH):
    with Image.open(IMAGE_PATH) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

image_b64 = encode_image_to_base64(IMAGE_PATH)

# --- Combine everything into the prompt ---
final_prompt = f"""{prompt}

Transcript:
{transcript}
"""

# --- Send the request to the Ollama model ---
response = ollama.chat(
    model=MODEL_NAME,  # change if your model name is different
    messages=[
        {
            "role": "user",
            "content": final_prompt,
            "images": [image_b64],  # 👈 include image here
        }
    ],
    format="json"  # 👈 ensures JSON output if your prompt expects it
)

# --- Print the structured response ---
print(response["message"]["content"])
