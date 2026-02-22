import json
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, send_from_directory
from flask_sock import Sock  # type: ignore

from utils.fetch_model import model

app = Flask(__name__)
sock = Sock(app)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _normalize_frame(frame: Any) -> np.ndarray:
    array_frame = np.asarray(frame, dtype=np.uint8)

    if array_frame.ndim == 3 and array_frame.shape[-1] == 3:
        array_frame = cv2.cvtColor(
            array_frame, cv2.COLOR_RGB2GRAY)  # type: ignore

    if array_frame.ndim != 2:
        raise ValueError(
            "Each frame must be a 2D grayscale image or 3-channel RGB image"
        )

    resized = cv2.resize(array_frame, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


def predict(frames: list[Any]) -> int:
    if len(frames) != 4:
        raise ValueError("Exactly 4 frames are required")

    normalized_frames = [_normalize_frame(frame) for frame in frames]

    obs = np.stack(normalized_frames, axis=-1).astype(np.uint8)
    obs = np.expand_dims(obs, axis=0)
    obs_transposed = np.transpose(obs, (0, 3, 1, 2))

    action = model.predict(obs_transposed)
    return int(np.asarray(action).flatten()[0])


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify({"status": "ok", "service": "flappy-bird-inference"}), 200


@app.get("/assets/<path:filename>")
def get_asset(filename: str):
    return send_from_directory(ASSETS_DIR, filename)


@sock.route("/predict")
def predict_ws(ws) -> None:
    frame_queue: deque[Any] = deque(maxlen=4)
    step_count = 0
    game_count = 0
    ACTION_REPEAT = 4

    while True:
        raw_message = ws.receive()
        if raw_message is None:
            break

        try:
            payload = json.loads(raw_message)

            if payload.get("reset"):
                game_count += 1
                print(f"New Game #{game_count}")
                frame_queue.clear()
                step_count = 0
                continue

            frame = payload.get("frame")
            if frame is None:
                continue

            step_count += 1
            if len(frame_queue) == 0:
                for _ in range(4):
                    frame_queue.append(frame)
            else:
                frame_queue.append(frame)

            print(f"Step Count: {step_count}")
            action = predict(list(frame_queue))
            ws.send(json.dumps({"action": action}))
            print(f"Action: {action}")
        except Exception as exc:
            ws.send(json.dumps({"error": str(exc)}))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
