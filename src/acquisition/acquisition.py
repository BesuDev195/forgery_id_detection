import cv2

def load_image(path, resize=None):
    """
    Load an image from disk.

    Args:
        path (str): Path to image
        resize (tuple): (width, height) to resize image

    Returns:
        img: Loaded image (BGR format)
    """
    
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Image not found at path: {path}")

    # Resize if specified
    if resize:
        img = cv2.resize(img, resize)

    return img