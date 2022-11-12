from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/select_pollen", methods=["POST"])
def select_pollen():
    image = request.files['file']

    return jsonify({"filename": image.filename})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
