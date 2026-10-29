"""Run from the repository or portable source ZIP without an editable install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helmetlab.kit_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
