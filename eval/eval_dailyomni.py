"""Score Daily-Omni predictions; invalid responses count as incorrect."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.data import load_items, parse_choice  # noqa: E402


def score(paths, data):
    gold = {item["id"]: item["answer"] for item in load_items(data)}
    seen = set()
    correct = invalid = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = str(row["id"])
                if key in seen or key not in gold:
                    raise ValueError(f"Duplicate or unknown sample id: {key}")
                seen.add(key)
                pred = parse_choice(row["response"])
                invalid += int(pred is None)
                correct += int(pred == gold[key])
    return dict(
        accuracy=100 * correct / len(seen) if seen else 0.0, correct=correct, evaluated=len(seen),
        expected=len(gold), missing=len(gold) - len(seen), invalid=invalid,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", help="one result file or disjoint shards of one run")
    parser.add_argument("--data", type=Path, default=ROOT / "json/dailyomni.json")
    args = parser.parse_args()
    print(json.dumps(score(args.predictions, args.data), indent=2))
