import numpy as np
import cv2 as cv

from pathlib import Path

path_to_models = Path(__file__).parent.resolve() / "models"

reference_pixels_per_micron = 10

def decode_b64_img(uri):
    # encoded_data = uri.split(',')[1]
    # arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    arr = np.frombuffer(uri, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    img = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    return img

def generate_crops(img, crop_locations):
    crops = []
    for crop_location in crop_locations:
        crop = img[crop_location[1]:crop_location[3], crop_location[0]:crop_location[2]]
        crops.append(crop)
    return crops