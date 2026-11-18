"""Evaluate rules / llm / hybrid on sample_data/labeled_cases.jsonl. Run by hand.

    env -u PYTHONPATH ./venv/bin/python scripts/evaluate.py --mode rules
    env -u PYTHONPATH ./venv/bin/python scripts/evaluate.py --mode llm
    env -u PYTHONPATH ./venv/bin/python scripts/evaluate.py --mode hybrid
    env -u PYTHONPATH ./venv/bin/python scripts/evaluate.py --summarize
    (add --fake to test the script itself with no key and no network)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so 'apidoctor' is importable

from apidoctor.evaluation import main

if __name__ == "__main__":
    sys.exit(main())
