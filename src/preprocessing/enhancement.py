import cv2

def enhance_image(img):
    """
    Enhance image for forgery detection.

    Steps:
    1. Convert to grayscale
    2. Apply noise reduction
    3. Improve contrast (important for text regions)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction (preserves edges better than Gaussian)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    return enhanced