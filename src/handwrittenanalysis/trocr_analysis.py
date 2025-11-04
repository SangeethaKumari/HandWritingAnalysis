from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
import requests
import torch
import cv2
import numpy as np

# load image from the IAM database
#url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
#image1 = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image_path = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/english_handwritten.jpg"

# Open and convert the image to RGB
image = Image.open(image_path).convert("RGB")

# Enhanced image preprocessing for better OCR accuracy
def preprocess_image(image):
    """Enhance image for better OCR accuracy"""
    # Convert PIL to numpy for OpenCV processing
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Resize if too small (helps with small text)
    h, w = gray.shape
    if min(h, w) < 300:
        scale = 300 / min(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    
    # Sharpen
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast, -1, kernel_sharpen)
    
    # Convert back to PIL Image
    enhanced_image = Image.fromarray(sharpened).convert("RGB")
    
    # Additional PIL enhancements
    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(1.2)  # Increase contrast by 20%
    
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(1.1)  # Slight sharpening
    
    return enhanced_image

# Preprocess the image
print("Preprocessing image...")
enhanced_image = preprocess_image(image)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use LARGE model for better accuracy (base is less accurate)
print("Loading TrOCR-large-handwritten model (more accurate)...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)
model.eval()  # Set model to evaluation mode

pixel_values = processor(images=enhanced_image, return_tensors="pt").pixel_values.to(device)

print("Generating text with beam search...")
with torch.no_grad():
    # Use beam search for better accuracy
    generated_ids = model.generate(
        pixel_values,
        max_length=512,
        num_beams=5,  # Beam search for better accuracy
        early_stopping=True,
        no_repeat_ngram_size=3  # Avoid repeating phrases
    )

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n" + "="*70)
print("Generated text:")
print("="*70)
print(generated_text)
print("="*70)

# Optional: Try with original image too (sometimes preprocessing can hurt)
print("\n\nTrying with original image (no preprocessing)...")
pixel_values_orig = processor(images=image, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    generated_ids_orig = model.generate(
        pixel_values_orig,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
generated_text_orig = processor.batch_decode(generated_ids_orig, skip_special_tokens=True)[0]

print("="*70)
print("Result with original image:")
print("="*70)
print(generated_text_orig)
print("="*70)