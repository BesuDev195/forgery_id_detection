import cv2

def apply_morphology(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return morph