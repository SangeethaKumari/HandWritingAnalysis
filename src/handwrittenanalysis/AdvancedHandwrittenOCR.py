"""
Advanced Handwritten Text OCR for English and Hindi (24GB GPU-Optimized)
Uses state-of-the-art models: TrOCR (Transformer-based) and PaddleOCR

Requirements:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers paddlepaddle-gpu paddleocr
pip install pillow opencv-python numpy easyocr
pip install accelerate bitsandbytes  # For model optimization
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from paddleocr import PaddleOCR
import easyocr
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Dict, Union, Tuple
import time

class AdvancedHandwrittenOCR:
    """
    Advanced OCR system combining multiple state-of-the-art models:
    - TrOCR: Microsoft's transformer-based OCR (best for English handwriting)
    - PaddleOCR: Baidu's industrial-grade OCR (excellent for Hindi & multi-language)
    - EasyOCR: Backup general-purpose OCR
    """
    
    def __init__(self, languages=['en', 'hi'], use_trocr=True, use_paddle=True, use_easy=False):
        """
        Initialize advanced OCR system
        
        Args:
            languages: List of language codes ['en', 'hi']
            use_trocr: Use Microsoft TrOCR (best for English handwriting)
            use_paddle: Use PaddleOCR (best for Hindi and mixed languages)
            use_easy: Use EasyOCR as backup
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            print("=" * 70)
            print(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"⚡ CUDA Version: {torch.version.cuda}")
            print(f"🔥 cuDNN Enabled: {torch.backends.cudnn.enabled}")
            print("=" * 70)
        else:
            print("⚠️  GPU not available, using CPU (will be slower)")
        
        self.models = {}
       
        # Initialize TrOCR (Best for English handwriting)
        if use_trocr and 'en' in languages:
            print("\n📥 Loading TrOCR (Transformer-based OCR for handwriting)...")
            try:
                # Using the handwritten model
                self.trocr_processor = TrOCRProcessor.from_pretrained(
                    'microsoft/trocr-large-handwritten'
                )
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    'microsoft/trocr-large-handwritten'
                ).to(self.device)
                self.trocr_model.eval()
                self.models['trocr'] = True
                print("✓ TrOCR loaded successfully!")
            except Exception as e:
                print(f"❌ TrOCR loading failed: {e}")
                self.models['trocr'] = False
       
        # Initialize PaddleOCR (Best for Hindi and multi-language)
        if use_paddle:
            print("\n📥 Loading PaddleOCR (Industrial-grade multi-language OCR)...")
            try:
                # PaddleOCR supports Hindi natively
                lang_map = {'en': 'en', 'hi': 'hindi'}
                paddle_langs = [lang_map.get(l, l) for l in languages]


                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=True,
                    show_log=False
                )
                    
                
                # For Hindi, initialize separate instance
                if 'hi' in languages:
                    self.paddle_ocr_hindi = PaddleOCR(
                        use_angle_cls=True,
                        lang='hindi',
                        use_gpu=True,
                        show_log=False
                    )
                    self.models['paddle_hindi'] = True
                
                self.models['paddle'] = True
                print("✓ PaddleOCR loaded successfully!")
            except Exception as e:
                print(f"❌ PaddleOCR loading failed: {e}")
                self.models['paddle'] = False
        
        # Initialize EasyOCR (General purpose backup)
        if use_easy:
            print("\n📥 Loading EasyOCR (General-purpose OCR)...")
            try:
                self.easy_reader = easyocr.Reader(languages, gpu=True)
                self.models['easy'] = True
                print("✓ EasyOCR loaded successfully!")
            except Exception as e:
                print(f"❌ EasyOCR loading failed: {e}")
                self.models['easy'] = False
        
        print("\n" + "=" * 70)
        print("✨ OCR System Ready!")
        print(f"Active Models: {', '.join([k for k, v in self.models.items() if v])}")
        print("=" * 70 + "\n")
    
    def preprocess_image(self, image_path: str, enhance: bool = True) -> np.ndarray:
        """
        Advanced image preprocessing for optimal OCR
        
        Args:
            image_path: Path to image
            enhance: Apply advanced enhancement techniques
            
        Returns:
            Preprocessed image
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        if enhance:
            # Resize if too small (helps with small text)
            h, w = gray.shape
            if min(h, w) < 300:
                scale = 300 / min(h, w)
                gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            
            # Denoise - use more aggressive denoising for handwriting
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Increase contrast using CLAHE (better for handwritten text)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            contrast = clahe.apply(denoised)
            
            # Sharpen the image to improve character clarity
            kernel_sharpen = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
            sharpened = cv2.filter2D(contrast, -1, kernel_sharpen)
            
            # Use Otsu's thresholding for better binarization
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to connect broken characters (gentler)
            kernel = np.ones((1, 1), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return morph
        
        return gray
    
    def extract_with_trocr(self, image_path: str, roi: Tuple = None) -> str:
        """
        Extract text using Microsoft TrOCR (best for English handwriting)
        
        Args:
            image_path: Path to image
            roi: Region of interest (x, y, w, h)
            
        Returns:
            Extracted text
        """
        if not self.models.get('trocr', False):
            return ""
        
        image = Image.open(image_path).convert("RGB")
        
        # Crop to ROI if specified
        if roi:
            x, y, w, h = roi
            image = image.crop((x, y, x + w, y + h))
        
        # Enhance image before processing for better accuracy
        # Resize if too small
        width, height = image.size
        if min(width, height) < 224:
            scale = 224 / min(width, height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Process image
        pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate text with better parameters for accuracy
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(
                pixel_values,
                max_length=512,
                num_beams=5,  # Beam search for better accuracy
                early_stopping=True
            )
        
        generated_text = self.trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return generated_text
    
    def extract_with_paddle(self, image_path: str, language: str = 'en') -> List[Dict]:
        """
        Extract text using PaddleOCR (best for Hindi and mixed languages)
        
        Args:
            image_path: Path to image
            language: 'en' or 'hi'
            
        Returns:
            List of detected text with bounding boxes
        """
        if not self.models.get('paddle', False):
            return []
        
        # Select appropriate model
        ocr = self.paddle_ocr_hindi if language == 'hi' and self.models.get('paddle_hindi') else self.paddle_ocr
        
        # Run OCR - PaddleOCR uses ocr() method, not predict()
        results = ocr.ocr(image_path, cls=True)
        
        # Format results
        formatted = []
        if results and results[0]:
            for line in results[0]:
                if line:
                    bbox = line[0]
                    text_info = line[1]
                    if isinstance(text_info, tuple) and len(text_info) == 2:
                        text, confidence = text_info
                    else:
                        text = str(text_info)
                        confidence = 1.0
                    formatted.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        return formatted
    
    def extract_with_easy(self, image_path: str) -> List[Dict]:
        """
        Extract text using EasyOCR
        
        Args:
            image_path: Path to image
            
        Returns:
            List of detected text with bounding boxes
        """
        if not self.models.get('easy', False):
            return []
        
        processed_img = self.preprocess_image(image_path)
        results = self.easy_reader.readtext(processed_img)
        
        formatted = []
        for (bbox, text, confidence) in results:
            formatted.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        return formatted
    
    def extract_text(self, image_path: str, method: str = 'auto', language: str = 'en') -> Dict:
        """
        Extract text using the best available method
        
        Args:
            image_path: Path to image
            method: 'auto', 'trocr', 'paddle', or 'easy'
            language: 'en' or 'hi'
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        start_time = time.time()
        results = {'image': image_path, 'text': '', 'method': method, 'details': []}
        
        # Auto-select best method
        if method == 'auto':
            if language == 'hi' and self.models.get('paddle'):
                method = 'paddle'
            elif language == 'en' and self.models.get('trocr'):
                method = 'trocr'
            elif self.models.get('paddle'):
                method = 'paddle'
            elif self.models.get('easy'):
                method = 'easy'
        
        print(f"Processing with {method.upper()} method...")
        
        # Extract using selected method
        if method == 'trocr':
            text = self.extract_with_trocr(image_path)
            results['text'] = text
            results['details'] = [{'text': text, 'confidence': 1.0}]
        
        elif method == 'paddle':
            details = self.extract_with_paddle(image_path, language)
            results['details'] = details
            results['text'] = '\n'.join([d['text'] for d in details])
        
        elif method == 'easy':
            details = self.extract_with_easy(image_path)
            results['details'] = details
            results['text'] = '\n'.join([d['text'] for d in details])
        
        results['processing_time'] = time.time() - start_time
        
        print(f"✓ Completed in {results['processing_time']:.2f}s")
        
        return results
    
    def batch_extract(self, image_paths: List[str], method: str = 'auto', language: str = 'en') -> List[Dict]:
        """
        Process multiple images in batch (GPU optimized)
        
        Args:
            image_paths: List of image paths
            method: OCR method to use
            language: Target language
            
        Returns:
            List of results for each image
        """
        print(f"\n🚀 Batch processing {len(image_paths)} images...")
        start_time = time.time()
        
        results = []
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] {img_path}")
            result = self.extract_text(img_path, method, language)
            results.append(result)
        
        total_time = time.time() - start_time
        print(f"\n✨ Batch completed in {total_time:.2f}s")
        print(f"⚡ Average: {total_time/len(image_paths):.2f}s per image")
        
        return results
    
    def compare_methods(self, image_path: str, language: str = 'en') -> Dict:
        """
        Compare results from all available methods
        
        Args:
            image_path: Path to image
            language: Target language
            
        Returns:
            Comparison results
        """
        print(f"\n🔬 Comparing all methods for: {image_path}")
        print("=" * 70)
        
        comparison = {}
        
        if self.models.get('trocr') and language == 'en':
            print("\n[TrOCR]")
            result = self.extract_text(image_path, 'trocr', language)
            comparison['trocr'] = result
            print(f"Text: {result['text']}")
            print(f"Time: {result['processing_time']:.2f}s")
        
        if self.models.get('paddle'):
            print("\n[PaddleOCR]")
            result = self.extract_text(image_path, 'paddle', language)
            comparison['paddle'] = result
            print(f"Text: {result['text']}")
            print(f"Time: {result['processing_time']:.2f}s")
            print(f"Avg Confidence: {np.mean([d['confidence'] for d in result['details']]):.2f}")
        
        if self.models.get('easy'):
            print("\n[EasyOCR]")
            result = self.extract_text(image_path, 'easy', language)
            comparison['easy'] = result
            print(f"Text: {result['text']}")
            print(f"Time: {result['processing_time']:.2f}s")
        
        print("\n" + "=" * 70)
        return comparison


def main():
    """
    Demo of advanced OCR capabilities
    """
    print("\n" + "=" * 70)
    print("🎯 ADVANCED HANDWRITTEN OCR SYSTEM")
    print("=" * 70)
    
    # Initialize with all models (leveraging 24GB GPU)
    ocr = AdvancedHandwrittenOCR(
        languages=['en', 'hi'],
        use_trocr=True,   # Best for English handwriting - ENABLED for better accuracy
        use_paddle=True,  # Best for Hindi and industrial use
        use_easy=True    # Can enable if needed
    )
    
    # Example 1: Extract English handwriting with TrOCR
    print("\n" + "=" * 70)
    print("Example 1: English Handwriting with TrOCR")
    print("=" * 70)
    
    image_path = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/english_handwritten.jpg"
    
    try:
        result = ocr.extract_text(image_path, method='auto', language='en')
        print(f"\n📝 Extracted Text:\n{result['text']}")
        print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
    except FileNotFoundError:
        print(f"\n⚠️  Please provide image: {image_path}")
    """
    # Example 2: Extract Hindi text with PaddleOCR
    print("\n" + "=" * 70)
    print("Example 2: Hindi Handwriting with PaddleOCR")
    print("=" * 70)
    
    hindi_image = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/hindi_handwritten.jpg"
    try:
        result = ocr.extract_text(hindi_image, method='paddle', language='hi')
        print(f"\n📝 Extracted Text:\n{result['text']}")
        print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
        
        # Show confidence scores
        if result['details']:
            avg_conf = np.mean([d['confidence'] for d in result['details']])
            print(f"📊 Average Confidence: {avg_conf:.2%}")
    except FileNotFoundError:
        print(f"\n⚠️  Please provide image: {hindi_image}")
    """
    # Example 3: Batch processing (GPU shines here!)
    print("\n" + "=" * 70)
    print("Example 3: Batch Processing Multiple Images")
    print("=" * 70)
    
    batch_images = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    ]
    
    try:
        #results = ocr.batch_extract(batch_images, method='auto', language='en')
        results = ""
        for i, result in enumerate(results, 1):
            print(f"\n--- Image {i} ---")
            print(result['text'][:100] + "..." if len(result['text']) > 100 else result['text'])
    except FileNotFoundError:
        print("\n⚠️  Batch processing requires valid image paths")
    
    # Example 4: Compare all methods
    print("\n" + "=" * 70)
    print("Example 4: Method Comparison")
    print("=" * 70)
    
    try:
        comparison = ocr.compare_methods(image_path, language='en')
    except FileNotFoundError:
        print(f"\n⚠️  Please provide valid image path")


if __name__ == "__main__":
    main()