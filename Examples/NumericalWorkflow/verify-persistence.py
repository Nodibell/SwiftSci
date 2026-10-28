"""Exercise training, saved prediction and labeled evaluation across fresh CLI processes."""

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

    # Named columns may contain spaces and commas; unrelated text stays unselected.
    custom_features = ["temperature C", "pressure,kPa", "offset"]
    renamed = dict(zip(features + ["class"], custom_features + ["outcome"]))
    custom_input, custom_query = root / "custom.csv", root / "custom-query.csv"
    custom_rows = [{**{renamed[key]: value for key, value in row.items()}, "notes": "unused text"}
                   for row in sample]
    for path, columns in [(custom_input, ["outcome", "notes", *reversed(custom_features)]),
                          (custom_query, ["notes", *reversed(custom_features)])]:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(custom_rows)
    custom_output = root / "custom-output"
    options = ["train", "--input", custom_input, "--output", custom_output,
               "--target", "outcome", "--components", "1", "--learning-rate", "0.0625", "--epochs", "12"]
    for name in custom_features:
        options += ["--feature", name]
    run(*options)
    custom_model = custom_output / "model.json"
    custom_saved = custom_model.read_bytes()
    fitted = json.loads(custom_saved)
    assert fitted["featureNames"] == custom_features and len(fitted["pcaComponents"]) == 1
    settings = json.loads((custom_output / "report.json").read_text())["trainingOptions"]
    assert settings == dict(featureNames=custom_features, targetColumn="outcome",
                            components=1, learningRate=0.0625, epochs=12)

    # Independently calculate the requested classifier updates on the saved projection.
    scores = []
    for row in custom_rows:
        centered = []
        for index, name in enumerate(custom_features):
            value = float(row[name]) if row[name] else math.nan
            if math.isnan(value):
                value = fitted["imputationStatistics"][index]
            centered.append((value - fitted["scalingMean"][index]) / fitted["scalingStd"][index]
                            - fitted["pcaMean"][index])
        scores.append(math.fsum(a * b for a, b in zip(centered, fitted["pcaComponents"][0])))
    weight, bias = 0.0, 0.0
    for _ in range(12):
        residuals = [1 / (1 + math.exp(-(score * weight + bias))) - float(row["outcome"])
                     for score, row in zip(scores[:8], custom_rows[:8])]
        weight -= 0.0625 * math.fsum(error * score for error, score in zip(residuals, scores[:8])) / 8
        bias -= 0.0625 * math.fsum(residuals) / 8
    assert abs(fitted["weights"][0] - weight) <= 1e-12 + 1e-12 * abs(weight)
    assert abs(fitted["bias"] - bias) <= 1e-12 + 1e-12 * abs(bias)
    custom_predictions = root / "custom-predictions.csv"
    run("predict", "--model", custom_model, "--input", custom_query, "--output", custom_predictions)
    custom_results = rows(custom_predictions)
    assert len(custom_results) == len(sample)
    for index, row in enumerate(custom_results):
        expected = 1 / (1 + math.exp(-(scores[index] * weight + bias)))
        assert int(row["source_row"]) == index and int(row["predicted"]) == int(expected > 0.5)
        assert abs(float(row["probability_class_1"]) - expected) <= 1e-12 + 1e-12 * abs(expected)
    for invalid in [["--feature", custom_features[0]], ["--feature", "outcome"], ["--target", ""],
                    ["--components", "0"], ["--components", "4"], ["--learning-rate", "nan"],
                    ["--epochs", "-1"], ["--target", "missing"]]:
        run(*options, *invalid, success=False)
        assert custom_model.read_bytes() == custom_saved

    single_output = root / "single-feature"
    run("train", "--input", custom_input, "--output", single_output, "--feature", custom_features[0],
        "--target", "outcome", "--components", "1", "--epochs", "0")
    single_model = single_output / "model.json"
    initialized = json.loads(single_model.read_bytes())
    assert initialized["featureNames"] == custom_features[:1]
    assert initialized["weights"] == [0] and initialized["bias"] == 0
    run("predict", "--model", single_model, "--input", custom_query, "--output", custom_predictions)
    initialized_results = rows(custom_predictions)
    assert len(initialized_results) == len(sample)
    assert all(int(row["predicted"]) == 0 and float(row["probability_class_1"]) == 0.5
               for row in initialized_results)

    # An imperfect fixture distinguishes macro F1 from support-weighted F1.
    evaluation_input = root / "evaluation.csv"
    selected_rows = [11, 10, 9, 8, 11, 10]
    ground_truth = [0, 0, 0, 0, 1, 1]
    evaluation_rows = [{**custom_rows[index], "outcome": label}
                       for index, label in zip(selected_rows, ground_truth)]

    def write_evaluation(data):
        with evaluation_input.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["outcome", "notes", *reversed(custom_features)])
            writer.writeheader()
            writer.writerows(data)

    write_evaluation(evaluation_rows)
    evaluation_output = root / "evaluation-output"
    evaluation_options = ["evaluate", "--model", custom_model, "--input", evaluation_input,
                          "--target", "outcome", "--output", evaluation_output]
    run(*evaluation_options)
    evaluation_report = evaluation_output / "evaluation.json"
    evaluation_csv = evaluation_output / "predictions.csv"
    summary = json.loads(evaluation_report.read_bytes())
    assert summary["schemaVersion"] == 1 and summary["targetColumn"] == "outcome"
    assert summary["querySourceRows"] == list(range(6))
    evaluation = summary["evaluation"]
    assert evaluation["classLabels"] == [0, 1] and evaluation["confusionMatrix"] == [[2, 2], [1, 1]]
    for name, expected in dict(accuracy=0.5, macroPrecision=0.5, macroRecall=0.5,
                               macroF1=17 / 35, weightedF1=18 / 35).items():
        assert math.isfinite(evaluation[name]) and abs(evaluation[name] - expected) <= 1e-12
    evaluated = rows(evaluation_csv)
    assert len(evaluated) == 6 and [int(row["actual"]) for row in evaluated] == ground_truth
    assert [int(row["predicted"]) for row in evaluated] == [0, 1, 1, 0, 0, 1]
    for index, row in enumerate(evaluated):
        assert int(row["source_row"]) == index
        expected = float(custom_results[selected_rows[index]]["probability_class_1"])
        assert abs(float(row["probability_class_1"]) - expected) <= 1e-12
    assert custom_model.read_bytes() == custom_saved
    original_report, original_csv = evaluation_report.read_bytes(), evaluation_csv.read_bytes()
    for invalid_label in [2, "NaN", ""]:
        write_evaluation([{**row, "outcome": invalid_label} if index == 0 else row
                          for index, row in enumerate(evaluation_rows)])
        run(*evaluation_options, success=False)
        assert evaluation_report.read_bytes() == original_report and evaluation_csv.read_bytes() == original_csv
    for invalid in [["--target", ""], ["--target", custom_features[0]], ["--input", custom_query]]:
        run(*evaluation_options, *invalid, success=False)
    write_evaluation(evaluation_rows)
    protected_directory = root / "protected-evaluation"
    protected_directory.mkdir()
    protected_input = protected_directory / "predictions.csv"
    protected_input.write_bytes(evaluation_input.read_bytes())
    protected_model = protected_directory / "evaluation.json"
    protected_model.write_bytes(custom_saved)
    run("evaluate", "--model", custom_model, "--input", protected_input, "--target", "outcome",
        "--output", protected_directory, success=False)
    run("evaluate", "--model", protected_model, "--input", evaluation_input, "--target", "outcome",
        "--output", protected_directory, success=False)
    assert protected_input.read_bytes() == evaluation_input.read_bytes()
    assert protected_model.read_bytes() == custom_saved
    write_evaluation([{**custom_rows[11], "outcome": 0}])
    run("evaluate", "--model", single_model, "--input", evaluation_input, "--target", "outcome",
        "--output", evaluation_output)
    single_evaluation = json.loads(evaluation_report.read_bytes())["evaluation"]
    assert single_evaluation["classLabels"] == [0] and single_evaluation["confusionMatrix"] == [[1]]
    assert single_evaluation["accuracy"] == single_evaluation["macroF1"] == 1

    artifact = json.loads(saved)
    artifact["bias"] += 0.1
    model.write_text(json.dumps(artifact), encoding="utf-8")
    oracle = subprocess.run([sys.executable, str(example / "verify.py"), str(trained)],
                            capture_output=True, text=True)
    assert oracle.returncode != 0, "Independent verifier accepted a changed model coefficient"

print(json.dumps({"status": "passed", "prediction_rows": len(sample),
                  "fresh_process_reload": True, "reordered_csv": True,
                  "configurable_training": True, "independent_classifier_updates": True,
                  "saved_model_evaluation": True, "independent_confusion_metrics": True,
                  "invalid_inputs_rejected": True, "oracle_negative_control": True}, indent=2))
