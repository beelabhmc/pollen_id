import numpy as np
import cv2 as cv

from pathlib import Path

path_to_models = Path(__file__).parent.resolve() / "models"

reference_pixels_per_micron = 15


classes_to_idx = {
    "Acmispon glaber": 0,
    "Amsinckia intermedia": 1,
    "Apiastrum angustifolium": 2,
    "Calystegia macrostegia": 3,
    "Camissonia bistorta": 4,
    "Carduus pycnocephalus": 5,
    "Centaurea melitensis": 6,
    "Corethrogyne filaginifolia": 7,
    "Croton setigerus": 8,
    "Encelia farinosa": 9,
    "Ericameria pinifolia": 10,
    "Eriogonum fasciculatum": 11,
    "Eriogonum gracile": 12,
    "Erodium Botrys": 13,
    "Erodium cicutarium": 14,
    "Heterotheca grandiflora": 15,
    "Hirschfeldia incana": 16,
    "Lepidospartum squamatum": 17,
    "Lessingia glandulifera": 18,
    "Malosma laurina": 19,
    "Marah Macrocarpa": 20,
    "Mirabilis laevis": 21,
    "Penstemon spectabilis": 22,
    "Phacelia distans": 23,
    "Rhus integrifolia": 24,
    "Ribes aureum": 25,
    "Salvia apiana": 26,
    "Sambucus nigra": 27,
    "Solanum umbelliferum": 28,
}

idx_to_classes = [
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
    img = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    return img

def generate_crops(img, crop_locations):
    crops = []
    for crop_location in crop_locations:
        crop = img[int(crop_location['y']):int(crop_location['y']+crop_location['h']), int(crop_location['x']):int(crop_location['x']+crop_location['h'])]
        crops.append(crop)
    return crops