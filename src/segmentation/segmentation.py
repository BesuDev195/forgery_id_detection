import cv2

def segment_image(img):
    edges = cv2.Canny(img, 100, 200)
    return edges