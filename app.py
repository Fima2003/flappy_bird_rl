import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, send_from_directory
from flask_sock import Sock  # type: ignore

from utils.results_tracker import record_game

app = Flask(__name__)
sock = Sock(app)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/version")
def version():
    return jsonify({"version": 0.1})


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify({"status": "ok", "service": "flappy-bird-inference"}), 200


@app.get("/assets/<path:filename>")
def get_asset(filename: str):
    return send_from_directory(ASSETS_DIR, filename)


@sock.route("/predict")
def predict_ws(ws) -> None:
    # Kept route as /predict to avoid breaking frontend compatibility
    # Now only used for tracking scores on reset
    while True:
        raw_message = ws.receive()
        if raw_message is None:
            break

        try:
            payload = json.loads(raw_message)

            if payload.get("reset"):
                score = payload.get("score", 0)
                record_game("web", score)

        except Exception as exc:
            ws.send(json.dumps({"error": str(exc)}))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
