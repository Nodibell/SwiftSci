"""Independently verify the bundled numerical example's JSON and CSV outputs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="Directory containing model.json, report.json and predictions.csv")
args = parser.parse_args()
report_path = args.output / "report.json"
csv_path = args.output / "predictions.csv"
report = json.loads(report_path.read_text(encoding="utf-8"))
model_path = args.output / "model.json"
model = json.loads(model_path.read_text(encoding="utf-8"))
errors = {}


def close(name, actual, expected):
    if isinstance(expected, list):
        assert len(actual) == len(expected), (name, "shape")
        for found, wanted in zip(actual, expected):
            close(name, found, wanted)
    else:
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (name, "numeric type")
        assert math.isfinite(actual) and math.isfinite(expected), (name, "nonfinite")
        difference = abs(actual - expected)
        errors[name] = max(errors.get(name, 0), difference)
        assert difference <= 1e-12 + 1e-12 * abs(expected), (name, actual, expected)


def dot(left, right):
    return math.fsum(x * y for x, y in zip(left, right))


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


training_ids = list(range(8))
query_ids = [11, 10, 9, 8]
raw = [[-4, -7, 7], [-3, -5, 7], [-2, None, 7], [-1, -1, 7],
       [1, 2, 7], [2, 3, 7], [3, 6, 7], [4, 8, 7],
       [-2.5, -4.5, 7], [2.5, 5, 7], [0.5, None, 7], [-0.5, 1, 7]]
targets = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]
assert report["schemaVersion"] == 2 and model["version"] == 1
assert report["trainingSourceRows"] == training_ids and report["querySourceRows"] == query_ids
assert model["featureNames"] == ["signal", "companion", "constant"]
assert set(model) == {"version", "featureNames", "imputationStatistics", "scalingMean", "scalingStd",
                      "pcaMean", "pcaComponents", "pcaVariance", "pcaTotalVariance", "weights", "bias"}

# Fit independent preprocessing parameters on training rows only.
replacements = []
for column in range(3):
    values = [raw[i][column] for i in training_ids if raw[i][column] is not None]
    replacements.append(math.fsum(values) / len(values))
imputed = [[replacements[j] if value is None else value for j, value in enumerate(row)] for row in raw]
means = [math.fsum(imputed[i][j] for i in training_ids) / 8 for j in range(3)]
deviations = [math.sqrt(math.fsum((imputed[i][j] - means[j]) ** 2 for i in training_ids) / 8) for j in range(3)]
deviations = [value if value >= 1e-12 else 1 for value in deviations]
scaled = [[(row[j] - means[j]) / deviations[j] for j in range(3)] for row in imputed]
training = [scaled[i] for i in training_ids]
query = [scaled[i] for i in query_ids]
close("imputation", model["imputationStatistics"], replacements)
close("scaling_mean", model["scalingMean"], means)
close("scaling_std", model["scalingStd"], deviations)

# Solve the two varying features' covariance eigensystem without a numerical library.
pca_mean = [math.fsum(row[j] for row in training) / 8 for j in range(3)]
centered = [[row[j] - pca_mean[j] for j in range(3)] for row in training]
aa = math.fsum(row[0] ** 2 for row in centered) / 7
bb = math.fsum(row[0] * row[1] for row in centered) / 7
cc = math.fsum(row[1] ** 2 for row in centered) / 7
discriminant = math.hypot(aa - cc, 2 * bb)
variances = [(aa + cc + discriminant) / 2, (aa + cc - discriminant) / 2]
assert bb != 0 and variances[0] > variances[1] > 0
vectors = []
for index, value in enumerate(variances):
    vector = [bb, value - aa, 0.0]
    length = math.sqrt(dot(vector, vector))
    vector = [x / length for x in vector]
    if dot(vector, model["pcaComponents"][index]) < 0:
        vector = [-x for x in vector]
    vectors.append(vector)
close("pca_mean", model["pcaMean"], pca_mean)
close("pca_variance", model["pcaVariance"], variances)
close("pca_total_variance", model["pcaTotalVariance"], aa + cc)
close("pca_components", model["pcaComponents"], vectors)
close("pca_orthogonality", dot(model["pcaComponents"][0], model["pcaComponents"][1]), 0)


def scores(rows):
    return [[dot([row[j] - pca_mean[j] for j in range(3)], vector) for vector in vectors] for row in rows]


training_scores, query_scores = scores(training), scores(query)

# Use independent full-batch binary logistic updates on the independently derived scores.
weights, bias = [0.0, 0.0], 0.0
for _ in range(64):
    residuals = [sigmoid(dot(row, weights) + bias) - targets[i] for i, row in enumerate(training_scores)]
    gradients = [math.fsum(error * row[j] for error, row in zip(residuals, training_scores)) / 8 for j in range(2)]
    weights = [weight - 0.125 * gradient for weight, gradient in zip(weights, gradients)]
    bias -= 0.125 * math.fsum(residuals) / 8
positive = [sigmoid(dot(row, weights) + bias) for row in query_scores]
probabilities = [[1 - value, value] for value in positive]
predictions = [int(value > 0.5) for value in positive]
close("logistic_weights", model["weights"], weights)
close("logistic_bias", model["bias"], bias)
accuracy = sum(prediction == targets[i] for prediction, i in zip(predictions, query_ids)) / len(query_ids)
close("accuracy", report["accuracy"], accuracy)

with csv_path.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    assert reader.fieldnames == ["source_row", "actual", "predicted", "probability_class_0", "probability_class_1"]
    rows = list(reader)
assert len(rows) == len(query_ids)
for index, row in enumerate(rows):
    assert int(row["source_row"]) == query_ids[index]
    assert int(row["actual"]) == targets[query_ids[index]]
    assert int(row["predicted"]) == predictions[index]
    for label in range(2):
        close("csv_probability", float(row[f"probability_class_{label}"]), probabilities[index][label])

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(dict(status="passed", training_rows=8, query_source_rows=query_ids,
                      tolerance="abs(error)<=1e-12+1e-12*abs(reference)",
                      maximum_absolute_errors=errors, predictions=predictions, accuracy=accuracy,
                      model_sha256=sha(model_path), report_sha256=sha(report_path), csv_sha256=sha(csv_path),
                      verifier_sha256=sha(Path(__file__))), indent=2))
