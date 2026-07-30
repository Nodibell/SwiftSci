# SwiftSci 2.4.0 Implementation Plan: Complete 14-Module DocC Documentation & SwiftNLP Engine

---

## 🎯 Strategic Goals & Architecture

**SwiftSci 2.4.0** delivers complete **100% DocC Documentation Coverage** across all **14 modules** in the SwiftSci ecosystem, paired with the newly implemented production-grade **`SwiftNLP`** engine and inter-module extensions.

Key architectural goals for Version 2.4.0:
1. **Module-by-Module DocC Completeness**: Every public protocol, struct, class, enum, method, property, and initializer across all 14 workspace targets has comprehensive triple-slash (`///`) DocC documentation comments with parameter (`- Parameter`), return (`- Returns`), and throws (`- Throws`) annotations.
2. **Dedicated `.docc` Catalog Articles**: Each of the 14 targets maintains a rich `.docc` bundle with topic groups, guides, code examples, and API navigation catalogs.
3. **NLTK-Inspired SwiftNLP Engine**: Full 2.4.0 NLP feature suite (`AppleWordTokenizer`, `SentenceTokenizer`, `RegexTokenizer`, `PorterStemmer`, `POSTagger`, `AppleLemmaTagger`, `AppleNamedEntityRecognizer`, `VADERSentimentAnalyzer`, `NLSentimentAnalyzer`, `AppleLanguageDetector`, `AppleNLEmbedding`, `MultinomialNaiveBayes`, `TextPipeline`).
4. **Deterministic Naming Rule for `Apple` Prefix**:
   - Apply `Apple` prefix **only** when plain names collide with Apple's `NaturalLanguage` framework (`AppleWordTokenizer`, `AppleLemmaTagger`, `AppleNamedEntityRecognizer`, `AppleNLEmbedding`).
   - Pure domain types without collision risk keep clean names (`SentenceTokenizer`, `POSTagger`, `PorterStemmer`, `VADERSentimentAnalyzer`).
5. **Zero-Warning Unified DocC Site**: Executing `./scripts/build_unified_docs.sh` generates a zero-warning unified static HTML documentation website for all 14 targets in `./docs`.

---

## 📅 Version 2.4.0 Phases Overview (Module by Module)

| Phase | Target Module | Scope & Objectives | DocC & API Goal |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **`SwiftDataFrame`** | DocC docstrings for columnar storage, CSV/JSON/Arrow IO, SIMD filtering, Radix sort, joins | 100% DocC coverage + `SwiftDataFrame.docc` catalog |
| **Phase 2** | **`SwiftStats`** | DocC docstrings for probability distributions, t-test, Chi-square, ANOVA, vDSP reductions | 100% DocC coverage + `SwiftStats.docc` catalog |
| **Phase 3** | **`SwiftPreprocessing`** | DocC docstrings for `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `PolynomialFeatures` | 100% DocC coverage + `SwiftPreprocessing.docc` catalog |
| **Phase 4** | **`SwiftML`** | DocC docstrings for Linear/Logistic Regression, Decision Trees, Random Forests, GBDT | 100% DocC coverage + `SwiftML.docc` catalog |
| **Phase 5** | **`SwiftCluster`** | DocC docstrings for KMeans, DBSCAN, PCA, IsolationForest, LocalOutlierFactor | 100% DocC coverage + `SwiftCluster.docc` catalog |
| **Phase 6** | **`SwiftNLP`** | DocC docstrings for Tokenizers, POS Tagger, PorterStemmer, VADER, NaiveBayes, `TextPipeline` | 100% DocC coverage + `SwiftNLP.docc` catalog |
| **Phase 7** | **`SwiftOptimize`** | DocC docstrings for L-BFGS, SGD, Adam optimizers, `GridSearchCV` hyperparameter tuning | 100% DocC coverage + `SwiftOptimize.docc` catalog |
| **Phase 8** | **`SwiftForecast`** | DocC docstrings for Exponential Smoothing, ARIMA, vDSP FFT & 1D Convolution decomposition | 100% DocC coverage + `SwiftForecast.docc` catalog |
| **Phase 9** | **`SwiftLLM`** | DocC docstrings for LLM inference, `LLMContextWindow`, BPE tokenizer, vector stores | 100% DocC coverage + `SwiftLLM.docc` catalog |
| **Phase 10** | **`SwiftExplain`** | DocC docstrings for `SHAPExplainer`, permutation feature importance, `TextExplainer` | 100% DocC coverage + `SwiftExplain.docc` catalog |
| **Phase 11** | **`SwiftVisualization`** | DocC docstrings for LinePlot, ScatterPlot, BarChart, Heatmap, `HTMLRenderer` | 100% DocC coverage + `SwiftVisualization.docc` catalog |
| **Phase 12** | **`SwiftVision`** | DocC docstrings for `ImageDataset`, Conv2D layers, U-Net segmentation, YOLOv8 detector | 100% DocC coverage + `SwiftVision.docc` catalog |
| **Phase 13** | **`SwiftDatabase`** | DocC docstrings for `PostgreSQLConnection`, `SQLiteConnection`, binary protocol parser | 100% DocC coverage + `SwiftDatabase.docc` catalog |
| **Phase 14** | **`SwiftAgent`** | DocC docstrings for `AgentEvaluator`, tool registry, RAG summary generation | 100% DocC coverage + `SwiftAgent.docc` catalog |
| **Phase 15** | **Unified DocC Site & Verification** | Run `./scripts/build_unified_docs.sh` & verify zero-warning build for all 14 targets | Verified zero-warning unified DocC site in `./docs` |

---

## 🔍 Detailed Plan by Module Phase

### Phase 1: `SwiftDataFrame` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftDataFrame/`
- **DocC Catalog**: `Sources/SwiftDataFrame/SwiftDataFrame.docc/`
- **Scope**:
  - Add missing parameter and return docstrings to `DataFrame.swift`, `TypedColumn.swift`, `CSVReader.swift`, `JSONReader.swift`, `DataFrame+Join.swift`, `DataFrame+Pivot.swift`, `DataFrame+Matrix.swift`.
  - Document SIMD bitmask filtering methods (`evaluatedIndices(where:)`, `gathered(atIndices:)`) and vDSP radix sorting.
  - Update `SwiftDataFrame.docc` catalog topic groups and tutorial guides.

---

### Phase 2: `SwiftStats` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftStats/`
- **DocC Catalog**: `Sources/SwiftStats/SwiftStats.docc/`
- **Scope**:
  - Add missing docstrings for `NormalDistribution`, `TDistribution`, `ChiSquareDistribution`, `FDistribution`, `PoissonDistribution`.
  - Document hypothesis testing methods (`tTest`, `chiSquareTest`, `anova`) and vDSP summary statistics reductions (`mean`, `variance`, `stdDev`, `skewness`, `kurtosis`).
  - Update `SwiftStats.docc` catalog.

---

### Phase 3: `SwiftPreprocessing` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftPreprocessing/`
- **DocC Catalog**: `Sources/SwiftPreprocessing/SwiftPreprocessing.docc/`
- **Scope**:
  - Document `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`.
  - Document categorical encoders (`OneHotEncoder`, `LabelEncoder`, `OrdinalEncoder`, `TargetEncoder`) and `PolynomialFeatures`.
  - Update `SwiftPreprocessing.docc` catalog.

---

### Phase 4: `SwiftML` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftML/`
- **DocC Catalog**: `Sources/SwiftML/SwiftML.docc/`
- **Scope**:
  - Document supervised learning models: `LinearRegression`, `RidgeRegression`, `LassoRegression`, `LogisticRegression`, `DecisionTree`, `RandomForestClassifier`, `GradientBoostedTreesClassifier`.
  - Document evaluation metrics (`accuracy`, `precision`, `recall`, `f1Score`, `rocAUC`, `mse`, `r2Score`) and model selection (`trainTestSplit`, `kFoldCrossValidation`).
  - Update `SwiftML.docc` catalog.

---

### Phase 5: `SwiftCluster` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftCluster/`
- **DocC Catalog**: `Sources/SwiftCluster/SwiftCluster.docc/`
- **Scope**:
  - Document clustering algorithms: `KMeans`, `DBSCAN`, `HierarchicalClustering`.
  - Document dimensionality reduction & anomaly detection: `PCA`, `IsolationForest`, `LocalOutlierFactor`.
  - Update `SwiftCluster.docc` catalog.

---

### Phase 6: `SwiftNLP` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftNLP/`
- **DocC Catalog**: `Sources/SwiftNLP/SwiftNLP.docc/`
- **Scope**:
  - Document tokenizers: `AppleWordTokenizer`, `SentenceTokenizer`, `RegexTokenizer`, `BPETokenizer`, `NGramTokenizer`.
  - Document linguistic processors & sentiment: `PorterStemmer`, `POSTagger`, `AppleLemmaTagger`, `AppleNamedEntityRecognizer`, `VADERSentimentAnalyzer`, `NLSentimentAnalyzer`.
  - Document classifiers & extensions: `MultinomialNaiveBayes`, `ComplementNaiveBayes`, `TextPipeline`, `DataFrame+NLP`, and inter-module extensions.
  - Update `SwiftNLP.docc` catalog overview and guides.

---

### Phase 7: `SwiftOptimize` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftOptimize/`
- **DocC Catalog**: `Sources/SwiftOptimize/SwiftOptimize.docc/`
- **Scope**:
  - Document mathematical optimization solvers: `LBFGS`, `SGD`, `Adam`, `NelderMead`, `GradientDescent`.
  - Document hyperparameter tuning: `GridSearchCV`, `RandomSearchCV`.
  - Update `SwiftOptimize.docc` catalog.

---

### Phase 8: `SwiftForecast` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftForecast/`
- **DocC Catalog**: `Sources/SwiftForecast/SwiftForecast.docc/`
- **Scope**:
  - Document time series models: `SimpleExponentialSmoothing`, `HoltsLinearSmoothing`, `HoltWintersSmoothing`, `ARIMA`.
  - Document decomposition and Fourier spectral engines: `TSDecomposition`, vDSP FFT frequency extraction, 1D FIR moving average convolution.
  - Update `SwiftForecast.docc` catalog.

---

### Phase 9: `SwiftLLM` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftLLM/`
- **DocC Catalog**: `Sources/SwiftLLM/SwiftLLM.docc/`
- **Scope**:
  - Document LLM inference engines, token generation, prompt templates, `LLMContextWindow`, vector store retrieval, and Metal/MLX accelerated matrix multiplication.
  - Update `SwiftLLM.docc` catalog.

---

### Phase 10: `SwiftExplain` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftExplain/`
- **DocC Catalog**: `Sources/SwiftExplain/SwiftExplain.docc/`
- **Scope**:
  - Document model explainability types: `SHAPExplainer` (Kernel SHAP, Tree SHAP), `PermutationImportance`, `TextExplainer` token importance scoring.
  - Update `SwiftExplain.docc` catalog.

---

### Phase 11: `SwiftVisualization` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftVisualization/`
- **DocC Catalog**: `Sources/SwiftVisualization/SwiftVisualization.docc/`
- **Scope**:
  - Document chart types: `LinePlot`, `ScatterPlot`, `BarChart`, `Histogram`, `Heatmap`, `BoxPlot`.
  - Document rendering backends: `HTMLRenderer`, SVG generator, theme customization.
  - Update `SwiftVisualization.docc` catalog.

---

### Phase 12: `SwiftVision` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftVision/`
- **DocC Catalog**: `Sources/SwiftVision/SwiftVision.docc/`
- **Scope**:
  - Document image datasets, image pre-processing (`resize`, `normalize`, `crop`), convolutional neural network layers (`Conv2D`, `BatchNorm`, `MaxPool2D`).
  - Document deep learning models: `UNetSegmentationModel`, `YOLOv8Detector` with Non-Maximum Suppression (NMS).
  - Update `SwiftVision.docc` catalog.

---

### Phase 13: `SwiftDatabase` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftDatabase/`
- **DocC Catalog**: `Sources/SwiftDatabase/SwiftDatabase.docc/`
- **Scope**:
  - Document database drivers: `PostgreSQLConnection`, `SQLiteConnection`, binary protocol query parsers, and zero-copy DataFrame ingestion.
  - Update `SwiftDatabase.docc` catalog.

---

### Phase 14: `SwiftAgent` Module DocC & API Polish

- **Target Directory**: `Sources/SwiftAgent/`
- **DocC Catalog**: `Sources/SwiftAgent/SwiftAgent.docc/`
- **Scope**:
  - Document AI agent execution: `AgentEvaluator`, `ToolRegistry`, tool parsing, prompt execution loops, and RAG document summary generation.
  - Update `SwiftAgent.docc` catalog.

---

### Phase 15: Unified DocC Site Generation & Verification

- **Script**: `./scripts/build_unified_docs.sh`
- **Target Output**: `./docs/`
- **Verification Steps**:
  1. Build DocC archives for all 14 targets into `.build/docc_tmp/`.
  2. Merge data, documentation, images, metadata, and index for all 14 targets into `./docs`.
  3. Execute `python3 scripts/merge_docc_indexes.py` to generate unified sidebar navigation.
  4. Verify zero compiler doc warnings across all 14 modules.
