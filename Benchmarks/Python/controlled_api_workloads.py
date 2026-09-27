"""CPU comparators for controlled models; contains no expected-answer imports."""
import numpy as np
from scipy.special import expit

OPERATIONS = {"linear-fixed-cpu", "logistic-fixed-cpu", "kmeans-one-cpu", "kalman-fixed-cpu"}


def prepare(payload):
    """Convert an already validated payload before timing."""
    result = dict(payload)
    if payload["operation"] in {"linear-fixed-cpu", "logistic-fixed-cpu"}:
        arrays = ["features", "weights"]
    elif payload["operation"] == "kmeans-one-cpu":
        arrays = ["features", "query"]
    elif payload["operation"] == "kalman-fixed-cpu":
        arrays = ["transition", "observation_matrix", "process_noise", "measurement_noise",
                  "initial_mean", "initial_covariance", "observations"]
    else:
        raise ValueError("Unknown controlled operation")
    for key in arrays:
        result[key] = np.asarray(payload[key], dtype=np.float64)
    return result


def execute(payload):
    """Materialize every contracted output inside timing, with fresh state per call."""
    operation = payload["operation"]
    if operation == "linear-fixed-cpu":
        predictions = payload["features"] @ payload["weights"] + payload["bias"]
        return np.concatenate(([payload["bias"]], payload["weights"], predictions))
    if operation == "logistic-fixed-cpu":
        def probabilities():
            positive = expit(payload["features"] @ payload["weights"] + payload["bias"])
            return np.column_stack((1-positive, positive))
        probability_rows = probabilities()
        # Both public Swift calls calculate probabilities, so mirror that work.
        labels = (probabilities()[:, 1] > 0.5).astype(np.float64)
        return np.concatenate(([payload["bias"]], payload["weights"], probability_rows.ravel(), labels))
    if operation == "kmeans-one-cpu":
        centroid = payload["features"].mean(axis=0, keepdims=True)
        differences = payload["features"][:, None, :] - centroid[None, :, :]
        distances = np.sum(differences*differences, axis=2)
        labels = np.argmin(distances, axis=1)
        query_differences = payload["query"][:, None, :] - centroid[None, :, :]
        query_labels = np.argmin(np.sum(query_differences*query_differences, axis=2), axis=1)
        inertia = np.sum(distances[np.arange(len(labels)), labels])
        return np.concatenate((centroid.ravel(), labels.astype(np.float64), query_labels.astype(np.float64), [inertia]))
    if operation == "kalman-fixed-cpu":
        transition, observation = payload["transition"], payload["observation_matrix"]
        process_noise, measurement_noise = payload["process_noise"], payload["measurement_noise"]
        mean, covariance = payload["initial_mean"].copy(), payload["initial_covariance"].copy()
        identity = np.eye(payload["state_size"])
        states = []
        for measurement in payload["observations"]:
            predicted_mean = transition @ mean
            predicted_covariance = transition @ covariance @ transition.T + process_noise
            innovation = measurement - observation @ predicted_mean
            innovation_covariance = observation @ predicted_covariance @ observation.T + measurement_noise
            gain = np.linalg.solve(innovation_covariance, (predicted_covariance @ observation.T).T).T
            mean = predicted_mean + gain @ innovation
            correction = identity - gain @ observation
            covariance = correction @ predicted_covariance @ correction.T + gain @ measurement_noise @ gain.T
            states.append(np.concatenate((mean, covariance.ravel())))
        next_mean = transition @ mean
        next_covariance = transition @ covariance @ transition.T + process_noise
        return np.concatenate(states + [np.concatenate((next_mean, next_covariance.ravel()))])
    raise ValueError("Unknown controlled operation")
