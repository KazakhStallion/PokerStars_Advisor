from pathlib import Path
from poker.ocr import run_simple_evaluator_from_json

if __name__ == "__main__":
    # Use the sample JSON produced by ocr
    json_path = Path(__file__).with_name("flop_state_20251206_153210.json")
    run_simple_evaluator_from_json(str(json_path))
