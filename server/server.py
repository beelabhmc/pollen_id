from flask import Flask, jsonify, request
from flask_cors import CORS
import simplejson as json

from api.utils import decode_b64_img, reference_pixels_per_micron, generate_crops
from api.select_pollen import find_pollen
from api.classify_pollen import classify

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/select_pollen", methods=["POST"])
def select_pollen():
    b64image = request.files['file']
    metadata = json.loads(request.form['metadata'])

    img = decode_b64_img(b64image.read())

    select_pollen = find_pollen(img, img_downscale=(metadata['pixels_per_micron'] / reference_pixels_per_micron) * 5)

    return jsonify({"filename": b64image.filename, "selected_pollen": select_pollen})

@app.route("/classify_pollen", methods=["POST"])
def classify_pollen():
    b64image = request.files['file']
    metadata = json.loads(request.form['metadata'])

    img = decode_b64_img(b64image.read())
    crops = generate_crops(img, metadata['crop_locations'])

    classifications = classify(crops)

    return jsonify({"filename": b64image.filename, "classified_pollen": classifications})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
