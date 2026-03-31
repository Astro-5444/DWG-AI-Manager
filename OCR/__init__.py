# OCR/__init__.py
"""
OCR Package for DWG file processing
Includes PDF rendering, YOLO detection, and Tesseract OCR
"""

# Import main functions for easy access
try:
    from .pdf_ocr_pipeline import process_pdf
    from .pdf_renderer import render_pdf_dual
    from .yolo_inference import predict_image
    from .ocr_extractor import extract_text
    
    __all__ = [
        'process_pdf',
        'render_pdf_dual', 
        'predict_image',
        'extract_text'
    ]
    
    # Flag indicating OCR is available
    OCR_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: OCR module components not fully available: {e}")
    OCR_AVAILABLE = False
    __all__ = []