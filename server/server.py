from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/select_pollen", methods=["POST"])
def hello_world():
    data = request.get_json()
    print(data)
    return jsonify({"message": "Hello, World!"})


if __name__ == "__main__":
    # app.debug=True
    app.run(host="0.0.0.0", port=8000)
