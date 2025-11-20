import easyocr
from PIL import Image
import numpy as np

def extract_handwritten_text(image_path):
    """
    Extract handwritten text from a whiteboard image
    
    Args:
        image_path: Path to the image file
    
    Returns:
        List of detected text with their locations and confidence scores
    """
    # Initialize EasyOCR reader for English
    # Set gpu=True if you have CUDA-enabled GPU for faster processing
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Read the image
    image = Image.open(image_path)
    
    # Convert PIL Image to numpy array for EasyOCR
    image_np = np.array(image)
    
    # Perform text detection and recognition
    # detail=1 returns bounding box coordinates along with text
    results = reader.readtext(image_np, detail=1)
    
    # Print results in a formatted way
    print("=" * 80)
    print("EXTRACTED TEXT FROM WHITEBOARD")
    print("=" * 80)
    print()
    
    all_text = []
    for idx, (bbox, text, confidence) in enumerate(results, 1):
        print(f"[{idx}] Text: {text}")
        print(f"    Confidence: {confidence:.2%}")
        print(f"    Location: {bbox[0]} -> {bbox[2]}")
        print()
        all_text.append(text)
    
    print("=" * 80)
    print("COMBINED TEXT:")
    print("=" * 80)
    print(" ".join(all_text))
    print()
    
    return results

# Main execution
if __name__ == "__main__":
    # Update this path to your image location
    image_path_1 = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/handwrittenanalysis/asif_notes.jpeg"
    image_path_2 = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/english_handwrittten.jpg"

    try:
        results = extract_handwritten_text(image_path_1)
        
        # Optional: Save results to a text file
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write("Extracted Text from Whiteboard\n")
            f.write("=" * 50 + "\n\n")
            for bbox, text, confidence in results:
                f.write(f"{text} (confidence: {confidence:.2%})\n")
        
        print(f"✓ Results saved to 'extracted_text.txt'")
        print(f"✓ Total text segments detected: {len(results)}")
        
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path_1}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error: {e}")