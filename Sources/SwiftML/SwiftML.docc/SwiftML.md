# ``SwiftML``

Machine Learning Framework for Swift on Apple Silicon.

## Overview

`SwiftML` provides vectorized linear, logistic, decision tree, random forest, gradient boosted trees, MLP neural networks, and multi-label classification algorithms leveraging Apple Accelerate (`vDSP`, `BLAS`, `LAPACK`).

## Topics

### Protocols & Core
- ``Estimator``
- ``Predictor``
- ``PredictionResult``
- ``EvaluationReport``
- ``FeatureSchema``

### Supervised Learning
- ``LinearRegression``
- ``LogisticRegression``
- ``DecisionTreeClassifier``
- ``RandomForestClassifier``
- ``GradientBoostedTreesClassifier``
- ``MLPClassifier``
- ``OneVsRestClassifier``
