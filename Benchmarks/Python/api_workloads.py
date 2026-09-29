"""CPU PCA and multinomial Naive Bayes comparators, separate from reference generation."""

import numpy as np
from scipy.special import logsumexp

OPERATIONS = {"pca-cpu", "multinomial-nb-cpu"}


def prepare(payload):
    """Convert an input accepted by numerical_fixtures.validate_input before timing."""
    result = dict(payload)
    result["features"] = np.asarray(payload["features"], dtype=np.float64)
    result["query"] = np.asarray(payload["query"], dtype=np.float64)
    if payload["operation"] == "multinomial-nb-cpu":
        result["targets"] = np.asarray(payload["targets"], dtype=np.float64)
    return result


def execute(payload):
    """Call inside timing, including all fitted state and numeric output materialization."""
    x, query = payload['features'],payload['query']
    if payload['operation'] == 'pca-cpu':
        k = payload['n_components']
        mean = x.mean(axis=0)
        centered = x-mean
        covariance = centered.T @ centered / (len(x)-1)
        eigenvalues,eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:k]
        variance = eigenvalues[order]
        components = eigenvectors[:,order].T
        ratios = variance / np.trace(covariance)
        scores = centered @ components.T
        query_scores = (query-mean) @ components.T
        projectors = np.einsum('ki,kj->kij',components,components)
        return np.concatenate((mean,variance,ratios,projectors.ravel(),
                               (scores@scores.T).ravel(),(query_scores@query_scores.T).ravel(),
                               (scores@query_scores.T).ravel()))
    if payload['operation'] == 'multinomial-nb-cpu':
        classes,counts = np.unique(payload['targets'],return_counts=True)
        priors = counts/len(x)
        feature_counts = np.vstack([x[payload['targets'] == label].sum(axis=0) for label in classes])
        smoothed = feature_counts+payload['alpha']
        log_features = np.log(smoothed/smoothed.sum(axis=1,keepdims=True))
        def probabilities():
            log_joint = query@log_features.T+np.log(priors)
            return np.exp(log_joint-logsumexp(log_joint,axis=1,keepdims=True))
        probs = probabilities()
        # Swift's public predict calls predictProbability again; mirror that work.
        indices = probabilities().argmax(axis=1)
        return np.concatenate((classes,probs.ravel(),indices.astype(np.float64)))
    raise ValueError('Unknown stage-two operation')
