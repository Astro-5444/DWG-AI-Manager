# OCR/yolo_crop.py
import cv2
import os
from pathlib import Path
from .yolo_inference import predict_image

# Map Class IDs to filenames
CLASS_MAPPING = {
    0: "Information_Box",
    1: "Floor_Plan_Box"
}

def detect_and_crop(low_res_path, high_res_path, output_dir=None, conf_threshold=0.5):
    """
    Detect boxes on low-res image using YOLO, then crop from high-res image.
    Supports Information Box and Floor Plan classes.
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(high_res_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if core images already exist
    fp_path = os.path.join(output_dir, "floor_plan.png")
    ib_path = os.path.join(output_dir, "Information_Box.png")
    if os.path.exists(fp_path) and os.path.exists(ib_path):
        return [fp_path, ib_path]
    
    # Load images
    low_img = cv2.imread(low_res_path)
    high_img = cv2.imread(high_res_path)
    
    if low_img is None:
        raise FileNotFoundError(f"Low-res image not found: {low_res_path}")
    if high_img is None:
        raise FileNotFoundError(f"High-res image not found: {high_res_path}")
    
    # Get dimensions for scaling
    h_low, w_low = low_img.shape[:2]
    h_high, w_high = high_img.shape[:2]
    
    # Calculate scale factors
    scale_x = w_high / w_low
    scale_y = h_high / h_low
    
    # Run YOLO detection on low-res image
    detections, _ = predict_image(
        low_img, 
        visualize=False, 
        conf_threshold=conf_threshold
    )
    
    cropped_paths = []
    
    # Logic to derive 'floor_plan' by cropping out the 'Information_Box'
    info_box_det = None
    for det in detections:
        if det.get('class') == 0: # Information_Box
            if info_box_det is None or det.get('conf') > info_box_det.get('conf'):
                info_box_det = det
                
    if info_box_det:
        ix1, iy1, ix2, iy2 = info_box_det['box']
        # Scale to high-res
        ix1_h, iy1_h = int(ix1 * scale_x), int(iy1 * scale_y)
        ix2_h, iy2_h = int(ix2 * scale_x), int(iy2 * scale_y)
        
        # Calculate possible floor plan areas (remaining rectangles)
        # 1. Left of info box
        # 2. Right of info box
        # 3. Top of info box
        # 4. Bottom of info box
        
        candidates = [
            ("left",   ix1_h * h_high,         (0, 0, ix1_h, h_high)),
            ("right",  (w_high - ix2_h) * h_high, (ix2_h, 0, w_high, h_high)),
            ("top",    w_high * iy1_h,         (0, 0, w_high, iy1_h)),
            ("bottom", w_high * (h_high - iy2_h), (0, iy2_h, w_high, h_high))
        ]
        
        # Pick the one with the largest area
        best_candidate = max(candidates, key=lambda x: x[1])
        _, _, (fx1, fy1, fx2, fy2) = best_candidate
        
        # Ensure coordinates are valid and not zero-area
        if fx2 > fx1 and fy2 > fy1:
            fp_crop = high_img[fy1:fy2, fx1:fx2]
            fp_filename = "floor_plan.png"
            fp_path = os.path.join(output_dir, fp_filename)
            cv2.imwrite(fp_path, fp_crop)
            cropped_paths.append(fp_path)
    
    # Crop each YOLO detection from high-res image as well
    for idx, det in enumerate(detections):
        class_id = det.get('class')
        class_name = CLASS_MAPPING.get(class_id, f"Detected_Box_{idx}")
        
        # Get box coordinates from low-res detection
        x1, y1, x2, y2 = det['box']
        
        # Scale coordinates to high-res image
        x1_high = int(x1 * scale_x)
        y1_high = int(y1 * scale_y)
        x2_high = int(x2 * scale_x)
        y2_high = int(y2 * scale_y)
        
        # Add small padding
        padding = 10
        x1_high = max(0, x1_high - padding)
        y1_high = max(0, y1_high - padding)
        x2_high = min(w_high, x2_high + padding)
        y2_high = min(h_high, y2_high + padding)
        
        # Crop from high-res image
        crop = high_img[y1_high:y2_high, x1_high:x2_high]
        
        # Save cropped image with specific name
        crop_filename = f"{class_name}.png"
        crop_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(crop_path, crop)
        
        if crop_path not in cropped_paths:
            cropped_paths.append(crop_path)
    
    return cropped_paths


def detect_and_crop_batch(low_res_path, high_res_path, output_dir=None, conf_threshold=0.5):
    """
    Optimized version for batch processing (same as above but with lower threshold).
    """
    return detect_and_crop(low_res_path, high_res_path, output_dir, conf_threshold)