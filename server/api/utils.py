import numpy as np
import cv2 as cv

from pathlib import Path

path_to_models = Path(__file__).parent.resolve() / "models"

reference_pixels_per_micron = 15


classes = [
    "Acmispon glaber",
    "Amsinckia intermedia",
    "Apiastrum angustifolium",
    "Calystegia macrostegia",
    "Camissonia bistorta",
    "Carduus pycnocephalus",
    "Centaurea melitensis",
    "Corethrogyne filaginifolia",
    "Croton setigerus",
    "Encelia farinosa",
    "Ericameria pinifolia",
    "Eriogonum fasciculatum",
    "Eriogonum gracile",
    "Erodium Botrys",
    "Erodium cicutarium",
    "Heterotheca grandiflora",
    "Hirschfeldia incana",
    "Lepidospartum squamatum",
    "Lessingia glandulifera",
    "Malosma laurina",
    "Marah Macrocarpa",
    "Mirabilis laevis",
    "Olea europaea",
    "Penstemon spectabilis",
    "Phacelia distans",
    "Rhus integrifolia",
    "Ribes aureum",
    "Salvia apiana",
    "Sambucus nigra",
    "Solanum umbelliferum",
]


def decode_b64_img(uri):
    # encoded_data = uri.split(',')[1]
    # arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    arr = np.frombuffer(uri, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def generate_crops(img, crop_locations):
    crops = []
    for crop_location in crop_locations:
        crop = img[
            int(crop_location["y"]) : int(crop_location["y"] + crop_location["h"]),
            int(crop_location["x"]) : int(crop_location["x"] + crop_location["h"]),
        ]
        crops.append(crop)
    return crops
