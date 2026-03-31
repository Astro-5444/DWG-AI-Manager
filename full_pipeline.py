import os
import sys
from DWG_Process.DWG_TO_PDF import plot_dwg_to_pdf
from OCR import process_pdf

def run_full_pipeline(dwg_path, output_root="output_pipeline"):
    """
    Runs the full pipeline: DWG -> PDF -> Images -> YOLO Crop -> OCR.
    
    Args:
        dwg_path (str): Path to the input DWG file.
        output_root (str): Root directory for outputs.
    """
    
    print(f"--- Starting Pipeline for: {dwg_path} ---")
    
    # 1. DWG to PDF
    print("[1/4] Converting DWG to PDF...")
    try:
        # We can specify a dedicated folder for the PDF output
        pdf_output_folder = os.path.join(output_root, "pdfs")
        pdf_path = plot_dwg_to_pdf(dwg_path, output_folder=pdf_output_folder)
        print(f"      PDF generated at: {pdf_path}")
    except Exception as e:
        print(f"Error during DWG to PDF conversion: {e}")
        return

    # 2. PDF to Images, 3. YOLO Crop, 4. OCR
    # process_pdf handles rendering, cropping, and OCR
    print("[2/4] Processing PDF (Render -> YOLO -> OCR)...")
    try:
        # We'll put images in a subfolder
        images_output_folder = os.path.join(output_root, "images")
        ocr_results = process_pdf(pdf_path, image_output_root=images_output_folder)
        print("      PDF processing complete.")
    except Exception as e:
        print(f"Error during PDF processing: {e}")
        return

    # Results
    print(f"\n[4/4] Pipeline Complete. Found {len(ocr_results)} regions.")
    for crop_path, text in ocr_results.items():
        print(f"\n--- Region: {os.path.basename(crop_path)} ---")
        print(f"Text: {text.strip()}")

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        dwg_file = sys.argv[1]
        run_full_pipeline(dwg_file)
    else:
        print("Usage: python full_pipeline.py <path_to_dwg_file>")
        # You can hardcode a path here for testing if you want
        # run_full_pipeline(r"C:\path\to\your.dwg")
