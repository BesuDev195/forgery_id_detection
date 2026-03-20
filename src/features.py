import numpy as np

def extract_features(img):
    edge_count = np.sum(img) / 255
    texture = np.var(img)
    return [edge_count, texture]