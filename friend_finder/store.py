import json
from pathlib import Path

DATA_PATH = Path("data/messages.json")

def load_messages():
    if not DATA_PATH.exists():
        return []
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))

def save_messages(messages):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
