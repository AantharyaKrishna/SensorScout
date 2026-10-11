import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "experiments.json"
)


def load_experiments():
    if not DATA_FILE.exists():
        return []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_experiment(result):
    experiments = load_experiments()

    record = {
        **result,
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    experiments.append(record)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = DATA_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(
            experiments,
            file,
            indent=2,
            allow_nan=False,
        )

    temporary_file.replace(DATA_FILE)

    return record["id"]

   