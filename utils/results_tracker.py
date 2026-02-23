import json
from pathlib import Path

RESULTS_FILE = Path(__file__).resolve().parent.parent / "results.json"


def record_game(platform: str, length: int):
    """
    platform: 'web' or 'python'
    length: integer representing the number of frames passed before death
    """

    # Ensure file exists
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            json.dump({
                "python_lengths": [],
                "web_lengths": [],
                "average_python_length": 0,
                "average_web_length": 0
            }, f, indent=4)

    # Read existing
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    key = f"{platform}_lengths"
    if key not in data:
        data[key] = []

    data[key].append(length)

    # Calculate average
    if data[key]:
        avg = sum(data[key]) / len(data[key])
        data[f"average_{platform}_length"] = round(avg, 2)

    # Write back
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=4)
