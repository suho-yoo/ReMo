"""WorldSense inference. Baseline and ReMo differ only by --remo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.run import parse_args, run  # noqa: E402

if __name__ == "__main__":
    run(parse_args("worldsense", "data/WorldSense"), "worldsense")
