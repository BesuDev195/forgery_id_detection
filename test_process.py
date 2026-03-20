import traceback
from src.model import process_image

print("Testing REAL:")
try:
    process_image('dataset/real/real.jpg')
    print("SUCCESS REAL")
except Exception:
    traceback.print_exc()

print("Testing FAKE:")
try:
    process_image('dataset/fake/fake.jpg')
    print("SUCCESS FAKE")
except Exception:
    traceback.print_exc()
