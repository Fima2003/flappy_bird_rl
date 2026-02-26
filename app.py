import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, send_from_directory, request
from flask_sock import Sock  # type: ignore
from dotenv import load_dotenv

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from google.cloud.sql.connector import Connector, IPTypes

from utils.results_tracker import record_game

load_dotenv()

app = Flask(__name__)
sock = Sock(app)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Initialize Cloud SQL Connector and SQLAlchemy
connector = Connector()


def getconn():
    return connector.connect(
        os.environ["INSTANCE_CONNECTION_NAME"],
        "pg8000",
        user=os.environ["DB_USER"],
        db=os.environ.get("DB_NAME", "postgres"),
        enable_iam_auth=os.environ.get(
            "DB_IAM_USER", "False").lower() == "true",
        ip_type=IPTypes.PUBLIC,
    )


engine = create_engine("postgresql+pg8000://", creator=getconn)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class GameHistory(Base):
    __tablename__ = "game_history"
    id = Column(Integer, primary_key=True, index=True)
    ai_score = Column(Integer, nullable=False)
    player_score = Column(Integer, nullable=False)
    winner = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


# Create table if it doesn't exist
Base.metadata.create_all(bind=engine)


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


@app.post("/log_score")
def log_score():
    data = request.json
    ai_score = data.get("ai_score", 0)
    player_score = data.get("player_score", 0)
    winner = data.get("winner", "Unknown")

    session = SessionLocal()
    try:
        record = GameHistory(
            ai_score=ai_score, player_score=player_score, winner=winner)
        session.add(record)
        session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.get("/history")
def get_history():
    session = SessionLocal()
    try:
        records = session.query(GameHistory).order_by(
            GameHistory.timestamp.desc()).limit(20).all()
        history = [
            {
                "id": r.id,
                "ai_score": r.ai_score,
                "player_score": r.player_score,
                "winner": r.winner,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None
            }
            for r in records
        ]
        return jsonify(history)
    finally:
        session.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
