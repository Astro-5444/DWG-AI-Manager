# yolo_inference.py

import os
import cv2
import numpy as np
from pathlib import Path

# ---------------------------
# Load trained model lazily
# ---------------------------
_base_dir = Path(__file__).parent.absolute()
_model_path = str(_base_dir / "best.pt")
_model = None
_device = None

def get_device():
    """Determine device lazily to avoid CUDA initialization on import."""
    import torch
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = 0
            try:
                device_name = torch.cuda.get_device_name(_device)
                print(f"[INFO] Using GPU for inference: {device_name}")
            except:
                print(f"[INFO] Using GPU (device 0)")
        else:
            _device = 'cpu'
            print("[INFO] GPU not found, using CPU (slower)")
    return _device

def get_model():
    """
    Lazy loader for the YOLO model.
    Ensures the model is loaded only when needed and only once per process.
    """
    from ultralytics import YOLO
    import torch
    import time
    global _model
    if _model is None:
        device = get_device()
        print(f"[INFO] Loading YOLO model from {_model_path}...")
        
        # Try loading with CUDA if selected
        if device != 'cpu':
            try:
                # Load with CUDA
                local_model = YOLO(_model_path)
                local_model.to('cuda')
                
                # Warm up model
                dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
                local_model.predict(source=dummy_img, device=device, verbose=False, imgsz=416, half=True)
                
                print(f"[INFO] Model loaded successfully on GPU")
                _model = local_model
                return _model
            except Exception as e:
                print(f"[WARNING] Failed to load YOLO on GPU: {e}")
                if "busy or unavailable" in str(e).lower():
                    print("[INFO] CUDA device busy, retrying once after 2 seconds...")
                    time.sleep(2)
                    try:
                        local_model = YOLO(_model_path)
                        local_model.to('cuda')
                        _model = local_model
                        print(f"[INFO] Model loaded successfully on GPU (retry)")
                        return _model
                    except:
                        pass
                
                print("[INFO] Falling back to CPU...")
                # Fallback to CPU state
                global _device
                _device = 'cpu'
                device = 'cpu'

        # Load on CPU if CUDA failed or wasn't selected
        try:
            local_model = YOLO(_model_path)
            local_model.to('cpu')
            
            # Warm up model on CPU
            dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
            local_model.predict(source=dummy_img, device='cpu', verbose=False, imgsz=416, half=False)
            
            print(f"[INFO] Model loaded successfully on CPU")
            _model = local_model
        except Exception as e:
            _model = None
            raise RuntimeError(f"Failed to load YOLO model on CPU: {e}")
            
    return _model


def predict_image(image, visualize=True, conf_threshold=0.6, img_size=512):
    """
    Run YOLO detection on a single image.
    
    Optimizations:
    - Lower confidence threshold (0.6 instead of 0.7)
    - Smaller image size (416 instead of 512) for faster processing
    - Half precision on GPU

    Args:
        image (np.ndarray or str): Image array (cv2) or path to image file.
        visualize (bool): Whether to draw boxes and show the image.
        conf_threshold (float): Confidence threshold for detection.
        img_size (int): Resize size for YOLO input.

    Returns:
        detections (list of dict): Each dict contains:
            - 'class': int, class ID
            - 'conf': float, confidence
            - 'box': [x1, y1, x2, y2]
        image (np.ndarray): Image with drawn boxes (if visualize=True)
    """
    # Load image if path is given
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image}")

    try:
        model = get_model()
        device = get_device()
        results = model.predict(
            source=image,
            device=device,
            imgsz=img_size,
            conf=conf_threshold,
            verbose=False,  # Disable verbose output
            half=(device != 'cpu')  # Use half precision on GPU
        )
    except Exception as e:
        raise RuntimeError(f"YOLO prediction failed: {e}")

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append({'class': cls, 'conf': conf, 'box': [x1, y1, x2, y2]})

            if visualize:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{cls} ({conf:.2f})"
                cv2.putText(image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if visualize:
        cv2.imshow("Detected Information Blocks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections, image


def predict_image_batch(images, visualize=False, conf_threshold=0.6, img_size=512, batch_size=8):
    """
    Run YOLO detection on multiple images in batch for better performance.
    
    Args:
        images (list): List of image arrays (np.ndarray) or paths
        visualize (bool): Whether to draw boxes
        conf_threshold (float): Confidence threshold
        img_size (int): Resize size for YOLO
        batch_size (int): Number of images to process at once
    
    Returns:
        all_detections (list of list): Detections for each image
        images (list): Processed images (if visualize=True)
    """
    # Load images if paths are given
    loaded_images = []
    for img in images:
        if isinstance(img, str):
            loaded_img = cv2.imread(img)
            if loaded_img is None:
                raise FileNotFoundError(f"Image not found: {img}")
            loaded_images.append(loaded_img)
        else:
            loaded_images.append(img)
    
    try:
        # Process in batches
        model = get_model()
        device = get_device()
        results = model.predict(
            source=loaded_images,
            device=device,
            imgsz=img_size,
            conf=conf_threshold,
            verbose=False,
            half=(device != 'cpu'),
            stream=True,  # Stream results for memory efficiency
            batch=batch_size
        )
    except Exception as e:
        raise RuntimeError(f"YOLO batch prediction failed: {e}")
    
    all_detections = []
    
    for idx, result in enumerate(results):
        img_detections = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            img_detections.append({'class': cls, 'conf': conf, 'box': [x1, y1, x2, y2]})
            
            if visualize and idx < len(loaded_images):
                cv2.rectangle(loaded_images[idx], (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{cls} ({conf:.2f})"
                cv2.putText(loaded_images[idx], label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        all_detections.append(img_detections)
    
    return all_detections, loaded_images


