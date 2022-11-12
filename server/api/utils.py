import numpy as np
import cv2 as cv

from pathlib import Path

path_to_models = Path(__file__).parent.resolve() / "models"

def decode_b64_img(uri):
    # encoded_data = uri.split(',')[1]
    # arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    arr = np.frombuffer(uri, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    return img
