"""Exercise saved workflows across fresh CLI processes using the bundled sample."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path, help="Built ScientificWorkflow executable")
args = parser.parse_args()
executable = args.executable.resolve()
example = Path(__file__).resolve().parent


def run(*arguments, success=True):
    process = subprocess.run([str(executable), *map(str, arguments)], cwd=root, capture_output=True, text=True)
    if success:
        assert process.returncode == 0, process.stderr
    else:
        assert process.returncode != 0, "Invalid request unexpectedly succeeded"
    return process


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


with tempfile.TemporaryDirectory(prefix="swiftsci-persistence-") as temporary:
    root = Path(temporary)
    trained = root / "trained"
    run("--output", trained)
    subprocess.run([sys.executable, str(example / "verify.py"), str(trained)],
                   check=True, capture_output=True, text=True)
    model = trained / "model.json"
    saved = model.read_bytes()
    sample = rows(example / "Data" / "sample.csv")
    features = json.loads(saved)["featureNames"]
    unlabeled = root / "unlabeled.csv"
    reordered = root / "reordered.csv"
    for path, order in [(unlabeled, features), (reordered, list(reversed(features)))]:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=order, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sample)
    predicted = root / "predicted.csv"
    reordered_predictions = root / "reordered-predictions.csv"
    run("predict", "--model", model, "--input", unlabeled, "--output", predicted)
    run("predict", "--model", model, "--input", reordered, "--output", reordered_predictions)
    predictions = rows(predicted)
    assert len(predictions) == len(sample)
    assert [int(row["source_row"]) for row in predictions] == list(range(len(sample)))
    assert "actual" not in predictions[0]
    assert predictions == rows(reordered_predictions)
    for heldout in rows(trained / "predictions.csv"):
        restored = predictions[int(heldout["source_row"])]
        assert restored["predicted"] == heldout["predicted"]
        for label in range(2):
            name = f"probability_class_{label}"
            actual, expected = float(restored[name]), float(heldout[name])
            assert math.isfinite(actual) and abs(actual - expected) <= 1e-12 + 1e-12 * abs(expected)
    assert model.read_bytes() == saved

    missing = root / "missing.csv"
    missing.write_text("signal,constant\n1,7\n", encoding="utf-8")
    run("predict", "--model", model, "--input", missing, "--output", predicted, success=False)
    for destination in [model, unlabeled]:
        original = destination.read_bytes()
        run("predict", "--model", model, "--input", unlabeled, "--output", destination, success=False)
        assert destination.read_bytes() == original
    corrupt = root / "corrupt.json"
    for changes in [{"version": 99}, {"scalingStd": [1, 0, 1]}, {"weights": [1]}]:
        artifact = json.loads(saved)
        artifact.update(changes)
        corrupt.write_text(json.dumps(artifact), encoding="utf-8")
        run("predict", "--model", corrupt, "--input", unlabeled, "--output", predicted, success=False)
    artifact = json.loads(saved)
    artifact["bias"] += 0.1
    model.write_text(json.dumps(artifact), encoding="utf-8")
    oracle = subprocess.run([sys.executable, str(example / "verify.py"), str(trained)],
                            capture_output=True, text=True)
    assert oracle.returncode != 0, "Independent verifier accepted a changed model coefficient"

print(json.dumps({"status": "passed", "prediction_rows": len(sample),
                  "fresh_process_reload": True, "reordered_csv": True,
                  "invalid_inputs_rejected": True, "oracle_negative_control": True}, indent=2))
