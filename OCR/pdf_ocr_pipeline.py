# pdf_ocr_pipeline.py

from .pdf_renderer import render_pdf_dual  
from .yolo_crop import detect_and_crop     
from .ocr_extractor import extract_text 


def process_pdf(pdf_file: str, image_output_root="images"):
    """
    Process a PDF: render, detect boxes, crop, run OCR.

    Args:
        pdf_file (str): Path to the PDF file.
        image_output_root (str): Root folder for generated images.

    Returns:
        dict: A dictionary with cropped image paths as keys and OCR text as values.
    """
    # --- 1. Render PDF into low-res and high-res images ---
    image_paths = render_pdf_dual(pdf_file, output_root=image_output_root)
    low_image_path = image_paths["low"]
    high_image_path = image_paths["high"]

    # --- 2. Run YOLO on low-res and crop boxes on high-res ---
    cropped_images = detect_and_crop(low_image_path, high_image_path, output_dir=None)

    # --- 3. Run OCR on each cropped image ---
    ocr_results = {}
    for crop_path in cropped_images:
        text = extract_text(crop_path)
        ocr_results[crop_path] = text

    return ocr_results
