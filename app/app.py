from flask import Flask, render_template, send_from_directory, abort
import os

app = Flask(__name__)

# Serve SAR TIFFs from your external folder
SAR_DIR = "/home/isaac/NASA_HC_SAR_2025/model_training/data/images"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sar/<path:filename>")
def serve_sar(filename):
    filepath = os.path.join(SAR_DIR, filename)
    if not os.path.exists(filepath):
        abort(404)
    return send_from_directory(SAR_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
