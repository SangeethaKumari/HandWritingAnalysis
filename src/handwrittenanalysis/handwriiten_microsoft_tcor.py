from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# load image from the IAM dataset
image_path_1 = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/handwritten_1.jpeg"
image_path_2 = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/english_handwritten.jpg"
image = Image.open(image_path_2).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)