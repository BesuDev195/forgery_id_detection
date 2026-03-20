from src.model import process_image
import numpy as np

f1 = process_image('dataset/real/real.jpg')
f2 = process_image('dataset/fake/fake.jpg')
print("Real features:", f1)
print("Fake features:", f2)

