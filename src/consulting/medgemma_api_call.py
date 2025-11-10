from transformers import pipeline
from PIL import Image
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",  # example for multimodal
    torch_dtype=torch.bfloat16,
    device="cuda"
)

image = Image.open("path_to_medical_image.jpg")
response = pipe([
    {
      "role": "system",
      "content": [{"type":"text","text":"You are an expert radiologist."}]
    },
    {
      "role": "user",
      "content": [
         {"type":"text","text":"Describe this scan."},
         {"type":"image","image": image}
      ]
    }
])
print(response[0]['generated_text'])
