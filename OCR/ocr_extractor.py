# ocr_extractor.py
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pytesseract
import os
import sys
from concurrent.futures import ThreadPoolExecutor


# Path to your local Tesseract executable
script_dir = os.path.dirname(os.path.abspath(__file__))
tesseract_dir = os.path.join(script_dir, "Tesseract")
tesseract_path = os.path.join(tesseract_dir, "tesseract.exe")
tessdata_dir = os.path.join(tesseract_dir, "tessdata")

# Set the Tesseract command path
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Set TESSDATA_PREFIX environment variable so Tesseract knows where to look for 'eng.traineddata'
# Standard Tesseract expects the directory containing the 'tessdata' folder
os.environ["TESSDATA_PREFIX"] = tesseract_dir


def extract_text_single(image_path):
    """Extract text from single image."""
    if not os.path.exists(image_path):
        return ""
    
    try:
        img = Image.open(image_path)
        # Use faster PSM mode and ensure TESSDATA_PREFIX is recognized via config as a fallback
        # Removed quotes around tessdata_dir to avoid "path"/eng.traineddata error
        custom_config = f'--oem 3 --psm 6 --tessdata-dir {tessdata_dir}'
        text = pytesseract.image_to_string(img, config=custom_config)
        return text
    except Exception as e:
        return f"Error: {str(e)}"


def extract_text_batch(image_paths, max_workers=10):
    """
    Extract text from multiple images in parallel using threading.
    
    Args:
        image_paths (list): List of image paths
        max_workers (int): Number of parallel OCR workers
    
    Returns:
        dict: {image_path: extracted_text}
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(extract_text_single, path): path 
                         for path in image_paths}
        
        for future in future_to_path:
            path = future_to_path[future]
            try:
                text = future.result()
                results[path] = text
            except Exception as e:
                results[path] = f"Error: {str(e)}"
    
    return results


def extract_text(image_path):
    """Backward compatible single image extraction."""
    return extract_text_single(image_path)

print(extract_text_single(r"C:\Users\kabdu\OneDrive\Desktop\AVIS\OCR_APR\output\images\DATA\25172  DGCL Capella Hotel\19. ICT\SLD\DG-NCD-405-0000-WME-DWG-IT-800-0000010_DWG(00)\high_DG-NCD-405-0000-WME-DWG-IT-800-0000010_DWG(00)_crop.png"))