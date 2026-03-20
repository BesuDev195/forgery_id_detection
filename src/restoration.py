import cv2
import numpy as np

def restore_image(img):
    """
    Applies image restoration techniques to improve forgery detection.
    
    Args:
        img: Input image (NumPy array)
        
    Returns:
        Restored image (NumPy array)
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply median blur to reduce noise while preserving edges
    restored = cv2.medianBlur(enhanced, 3)
    
    # Optional: Apply sharpening filter
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # restored = cv2.filter2D(restored, -1, kernel)
    
    return restored