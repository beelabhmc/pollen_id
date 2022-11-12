from flask import Flask, jsonify, request
from flask_cors import CORS

from api.utils import decode_b64_img
from api.select_pollen import find_pollen
from api.classify_pollen import classify_pollen

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/select_pollen", methods=["POST"])
def select_pollen():
    b64image = request.files['file']

    img = decode_b64_img(b64image.read())

    select_pollen = find_pollen(img)

    return jsonify({"filename": b64image.filename, "selected_pollen": select_pollen})

@app.route("/select_pollen", methods=["POST"])
def classify_pollen():
    b64image = request.files['file']
    crop_locations = request.files['crop_locations']

    img = decode_b64_img(b64image.read())
    print(crop_locations)

    return jsonify({"filename": b64image.filename, "classified_pollen": []})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
