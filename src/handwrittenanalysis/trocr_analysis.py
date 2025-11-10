from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
import requests
import torch
import cv2
import numpy as np

# load image from the IAM database
#url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
#image1 = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image_path = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/handwritten_1.jpeg"

# Open and convert the image to RGB
image = Image.open(image_path).convert("RGB")

# Enhanced image preprocessing for better OCR accuracy
def preprocess_image(image, aggressive=False):
    """Enhance image for better OCR accuracy"""
    # Convert PIL to numpy for OpenCV processing
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Resize if too small (helps with small text) - more aggressive for better character distinction
    h, w = gray.shape
    if min(h, w) < 400:  # Increased threshold
        scale = 400 / min(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # Denoise - lighter denoising to preserve character details
    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    
    # Increase contrast using CLAHE - more aggressive for better distinction
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    
    # Sharpen more aggressively to distinguish similar characters (L vs r)
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 11,-1],  # Stronger sharpening
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast, -1, kernel_sharpen)
    
    # Additional morphological operations to separate characters
    if aggressive:
        # Light dilation to separate touching characters
        kernel = np.ones((2, 1), np.uint8)
        sharpened = cv2.dilate(sharpened, kernel, iterations=1)
    
    # Convert back to PIL Image
    enhanced_image = Image.fromarray(sharpened).convert("RGB")
    
    # Additional PIL enhancements
    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(1.3)  # More contrast
    
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(1.2)  # More sharpening
    
    return enhanced_image

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use LARGE model for better accuracy (base is less accurate)
print("Loading TrOCR-large-handwritten model (more accurate)...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten').to(device)
model.eval()  # Set model to evaluation mode

# Try multiple preprocessing strategies
results = {}

# Strategy 1: Enhanced preprocessing
print("\n" + "="*70)
print("Strategy 1: Enhanced preprocessing")
print("="*70)
enhanced_image = preprocess_image(image, aggressive=False)
pixel_values = processor(images=enhanced_image, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    generated_ids = model.generate(
        pixel_values,
        max_length=512,
        num_beams=10,  # Increased beam width for better exploration
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=0.6  # Prefer shorter sequences (ML vs Mr.)
    )
results['enhanced'] = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Result: {results['enhanced']}")

# Strategy 2: Aggressive preprocessing
print("\n" + "="*70)
print("Strategy 2: Aggressive preprocessing")
print("="*70)
aggressive_image = preprocess_image(image, aggressive=True)
pixel_values_agg = processor(images=aggressive_image, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    generated_ids_agg = model.generate(
        pixel_values_agg,
        max_length=512,
        num_beams=10,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=0.6
    )
results['aggressive'] = processor.batch_decode(generated_ids_agg, skip_special_tokens=True)[0]
print(f"Result: {results['aggressive']}")

# Strategy 3: Original image (no preprocessing)
print("\n" + "="*70)
print("Strategy 3: Original image (no preprocessing)")
print("="*70)
pixel_values_orig = processor(images=image, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    generated_ids_orig = model.generate(
        pixel_values_orig,
        max_length=512,
        num_beams=10,  # Increased beam width
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=0.6
    )
results['original'] = processor.batch_decode(generated_ids_orig, skip_special_tokens=True)[0]
print(f"Result: {results['original']}")

# Strategy 4: Top-k sampling (alternative to beam search)
print("\n" + "="*70)
print("Strategy 4: Top-k sampling (alternative approach)")
print("="*70)
pixel_values_alt = processor(images=enhanced_image, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    generated_ids_alt = model.generate(
        pixel_values_alt,
        max_length=512,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=3  # Get multiple candidates
    )
alt_results = processor.batch_decode(generated_ids_alt, skip_special_tokens=True)
results['sampling'] = alt_results[0]
print(f"Top result: {alt_results[0]}")
if len(alt_results) > 1:
    print(f"Alternative 1: {alt_results[1]}")
    print(f"Alternative 2: {alt_results[2]}")

# Summary
print("\n" + "="*70)
print("SUMMARY - All Results:")
print("="*70)
for strategy, text in results.items():
    print(f"{strategy.upper():15}: {text}")
print("="*70)
print(f"\nBest guess (most likely): {results.get('enhanced', 'N/A')}")