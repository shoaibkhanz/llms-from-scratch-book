"""load data"""

from pathlib import Path


def read_data(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text


nif __name__ == "__main__":
    data = read_data(Path(__file__).parents[2] / "resources" / "verdict.txt")
