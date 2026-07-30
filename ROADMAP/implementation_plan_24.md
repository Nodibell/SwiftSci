# SwiftSci 2.4.0 Implementation Plan: Performant Swift NLP Engine (Apple NaturalLanguage Integration & NLTK-Inspired APIs)

---

## 🎯 Strategic Goals & Architecture

**SwiftSci 2.4.0** focuses on delivering a production-grade, highly performant Natural Language Processing framework for Swift (`SwiftNLP`), providing **NLTK-inspired APIs** for developer familiarity while deeply integrating Apple's hardware-accelerated **`NaturalLanguage`** and **`Accelerate`** frameworks.

Key architectural principles for `SwiftNLP 2.4.0`:
1. **NLTK-Inspired Functional API**: Familiar high-level interfaces for tokenization, stemming, lemmatization, sentiment analysis, entity extraction, and text classification. (Note: Focus is on practical NLP pipelines rather than replicating NLTK's heavy WordNet synset graph taxonomy).
2. **Apple `NaturalLanguage` Framework Integration**: Direct use of Apple's multi-lingual `NLTokenizer`, `NLTagger`, `NLLanguageRecognizer`, and `NLEmbedding` on macOS/iOS/visionOS.
3. **Deterministic Naming Rule for `Apple` Prefix**:
   - Apply the `Apple` prefix **only** when a plain name creates ambiguity or autocomplete collisions with Apple's `NaturalLanguage` framework (e.g., `AppleWordTokenizer`, `AppleLemmaTagger`, `AppleNamedEntityRecognizer`, `AppleNLEmbedding`).
   - Domain types with no collision risk use clean, un-prefixed names (`SentenceTokenizer`, `POSTagger`, `PorterStemmer`, `VADERSentimentAnalyzer`).
4. **Explicit Cross-Platform & Linux Compatibility**:
   - Components with pure Swift fallbacks (`PorterStemmer`, `VADERSentimentAnalyzer`, `MultinomialNaiveBayes`) run on all platforms (Apple + Linux).
   - Apple OS-bound features (`AppleNamedEntityRecognizer`, `AppleLanguageDetector`, `AppleNLEmbedding`) are guarded via `#if canImport(NaturalLanguage)` and throw `NLPError.unavailableOnPlatform` on Linux.
5. **Zero-External-Dependency Storage (VADER Lexicon)**: VADER sentiment valence lexicon (~200KB) is statically embedded as pre-sorted key-value arrays in `VADERLexicon.swift` (via a build helper script to ensure zero type-checker overhead during Swift compilation).
6. **Accelerate SIMD Vector Operations**: Accelerate (`vDSP_dotprD`, `vDSP_svesqD`) for high-speed embedding similarity calculations and sparse vectorization.
7. **Fluent `DataFrame` NLP Pipeline**: Zero-copy tokenization, sentiment scoring, entity extraction, and vectorization directly on `SwiftDataFrame` columns.

---

## 📅 Version 2.4.0 Phases Overview

| Phase | Target Feature / Component | Scope & Objectives | Key Platform Support |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Tokenizer & Text Segmentation Engine** | `AppleWordTokenizer`, `SentenceTokenizer`, `RegexTokenizer`, `NGramTokenizer` | Apple Native + Pure Swift |
| **Phase 2** | **Linguistic Processing: POS Tagging & Stemming/Lemmatization** | `POSTagger` (`NLTagger` + Rule fallback), `PorterStemmer`, `AppleLemmaTagger` | Apple Native + Pure Swift Fallback |
| **Phase 3** | **Named Entity Recognition (NER)** | `AppleNamedEntityRecognizer` for Person, Location, Org with character spans | macOS/iOS/visionOS (`.unavailableOnPlatform` on Linux) |
| **Phase 4** | **Dual Sentiment Analysis Engine** | `VADERSentimentAnalyzer` (static 200KB lexicon) & `NLSentimentAnalyzer` | All Platforms (VADER) / Apple (NLSentiment) |
| **Phase 5** | **Language Detection & Embedding Vector Search** | `AppleLanguageDetector` & `AppleNLEmbedding` with Accelerate SIMD Cosine Distance | Apple (`NLEmbedding`) + All Platforms (`WordEmbeddings`) |
| **Phase 6** | **Text Classification Engine (Naive Bayes Classifiers)** | `MultinomialNaiveBayes` & `ComplementNaiveBayes` with text pipeline integration | Pure Swift (All Platforms) |
| **Phase 7** | **Fluent `DataFrame + NLP` Extensions & Verification** | Complete column operations (`tokenizeColumn`, `extractEntities`) & Benchmark Suite | Complete test & empirical benchmark suite |

---

## 🔍 Detailed Plan by Phase

### Phase 1: Tokenizer & Text Segmentation Engines (`SwiftNLP`)

- **Target Files**:
  - `Sources/SwiftNLP/Core/Tokenizer.swift`
  - [NEW] `Sources/SwiftNLP/Core/AppleWordTokenizer.swift`
  - [NEW] `Sources/SwiftNLP/Core/SentenceTokenizer.swift`
  - [NEW] `Sources/SwiftNLP/Core/RegexTokenizer.swift`
- **Design Details**:
  - `AppleWordTokenizer` wraps Apple's `NLTokenizer(unit: .word)`. Uses `Apple` prefix to avoid collision with `NLTokenizer`.
  - `SentenceTokenizer` provides sentence boundary splitting using `NLTokenizer(unit: .sentence)` on Apple platforms and regex rule-based sentence splitting on Linux.
  - `RegexTokenizer` provides configurable regex token splitting (equivalent to NLTK's `RegexpTokenizer`).

---

### Phase 2: POS Tagging, Stemming & Lemmatization (`SwiftNLP`)

- **Target Files**:
  - [NEW] `Sources/SwiftNLP/Core/POSTagger.swift`
  - [NEW] `Sources/SwiftNLP/Core/PorterStemmer.swift`
  - [NEW] `Sources/SwiftNLP/Core/AppleLemmaTagger.swift`
- **Design Details**:
  - **`POSTagger`**: Uses `NLTagger(tagSchemes: [.lexicalClass])` on Apple platforms, with basic rule-based POS tagging fallback for Linux.
  - **`PorterStemmer`**: Pure Swift implementation of the classic Porter Stemming algorithm for suffix stripping (`running` → `run`). Works on all platforms.
  - **`AppleLemmaTagger`**: Uses `NLTagger(tagSchemes: [.lemma])` for dictionary canonical form extraction without requiring external 100MB+ WordNet database files.
  - *Explicit Scope Note*: WordNet synset taxonomy trees (hypernyms/hyponyms, Wu-Palmer distance) are explicitly non-goals for 2.4.0 to keep memory footprint light and zero-dependency.

---

### Phase 3: Named Entity Recognition (NER) (`SwiftNLP`)

- **Target Files**:
  - [NEW] `Sources/SwiftNLP/Core/AppleNamedEntityRecognizer.swift`
  - [NEW] `Sources/SwiftNLP/Models/NamedEntity.swift`
- **Design Details**:
  - Data model `NamedEntity` (`text: String`, `category: EntityCategory`, `range: Range<String.Index>`).
  - `AppleNamedEntityRecognizer` wraps `NLTagger(tagSchemes: [.nameTypeOrLexicalClass])` on Apple platforms.
  - **Linux Behavior**: On Linux (`#if !canImport(NaturalLanguage)`), calling entity recognition throws `NLPError.unavailableOnPlatform(feature: "Named Entity Recognition")` rather than failing silently.

---

### Phase 4: Dual Sentiment Analysis Engine (VADER & Native OS ML) (`SwiftNLP`)

- **Target Files**:
  - [NEW] `Sources/SwiftNLP/Sentiment/VADERSentimentAnalyzer.swift`
  - [NEW] `Sources/SwiftNLP/Sentiment/NLSentimentAnalyzer.swift`
  - [NEW] `Sources/SwiftNLP/Sentiment/VADERLexicon.swift`
  - [NEW] `Sources/SwiftNLP/Sentiment/SentimentScore.swift`
- **Design Details**:
  - **`VADERLexicon.swift`**: Generated pre-sorted array lookup or flat binary layout representing the ~200KB VADER valence lexicon to avoid Swift compiler dictionary literal type-checker slowness.
  - **`VADERSentimentAnalyzer`**: Pure Swift rule-based analyzer (ALL CAPS booster, exclamation marks, negations `not`/`never`, idioms). Runs everywhere (macOS, iOS, Linux).
  - **`NLSentimentAnalyzer`**: Wraps `NLTagger(tagSchemes: [.sentimentScore])` for Apple OS ML model sentiment analysis.

---

### Phase 5: Language Identification & `NLEmbedding` Vector Search (`SwiftNLP`)

- **Target Files**:
  - [NEW] `Sources/SwiftNLP/Core/AppleLanguageDetector.swift`
  - [NEW] `Sources/SwiftNLP/Core/AppleNLEmbedding.swift`
  - [MODIFY] `Sources/SwiftNLP/Core/WordEmbeddings.swift`
- **Design Details**:
  - **`AppleLanguageDetector`**: Wraps `NLLanguageRecognizer` on Apple platforms; throws `.unavailableOnPlatform` on Linux.
  - **`AppleNLEmbedding`**: Wraps Apple OS built-in word & sentence vector spaces (`NLEmbedding`).
  - **SIMD Vector Acceleration**: Accelerate `vDSP_dotprD` and `vDSP_svesqD` for batch embedding similarity matrix computation in `WordEmbeddings`. Performance will be empirically measured in Phase 7 benchmarks.

---

### Phase 6: Text Classification Engine (Naive Bayes Classifiers) (`SwiftNLP`)

- **Target Files**:
  - [NEW] `Sources/SwiftNLP/Classification/MultinomialNaiveBayes.swift`
  - [NEW] `Sources/SwiftNLP/Classification/ComplementNaiveBayes.swift`
  - [NEW] `Sources/SwiftNLP/Classification/TextClassifierPipeline.swift`
- **Design Details**:
  - Pure Swift implementations of `MultinomialNaiveBayes` and `ComplementNaiveBayes` (for imbalanced corpora).
  - Works seamlessly across Apple and Linux platforms.

---

### Phase 7: Fluent `DataFrame + NLP` API & Verification Suite (`SwiftNLP`)

- **Target Files**:
  - [MODIFY] [DataFrame+NLP.swift](file:///Users/oleksiichumak/Developer/Xcode.projects/SwiftSci/SwiftSci/Sources/SwiftNLP/Core/DataFrame+NLP.swift)
  - [NEW] `Tests/SwiftNLPTests/POSTaggerTests.swift`
  - [NEW] `Tests/SwiftNLPTests/VADERSentimentTests.swift`
  - [NEW] `Tests/SwiftNLPTests/AppleNamedEntityRecognizerTests.swift`
  - [NEW] `Tests/SwiftNLPTests/NaiveBayesClassifierTests.swift`
- **Design Details**:
  - `df.tokenizeColumn("text", targetColumn: "tokens", tokenizer: AppleWordTokenizer())`
  - `df.stemColumn("tokens", targetColumn: "stemmed")`
  - `df.analyzeSentiment(column: "text", targetColumn: "sentiment")`
  - Benchmark execution vs Python baseline script to record measured runtime performance.

---

## 🛠️ Implementation Dependencies & API Compatibility Table

| NLTK Feature / Capability | Apple Native API | SwiftSci 2.4.0 Type Name | Linux Support Status |
| :--- | :--- | :--- | :--- |
| `word_tokenize` | `NLTokenizer(.word)` | `AppleWordTokenizer` | Supported (Pure Swift Regex fallback) |
| `sent_tokenize` | `NLTokenizer(.sentence)` | `SentenceTokenizer` | Supported (Pure Swift Regex fallback) |
| `pos_tag` | `NLTagger(.lexicalClass)` | `POSTagger` | Apple Native + Rule Fallback |
| `PorterStemmer` | N/A | `PorterStemmer` | Pure Swift (All Platforms) |
| `WordNetLemmatizer` (Lemma only) | `NLTagger(.lemma)` | `AppleLemmaTagger` | Apple Native (Throws `.unavailableOnPlatform` on Linux) |
| WordNet Synset Graph | N/A | *Explicit Non-Goal in 2.4* | N/A |
| `ne_chunk` (NER) | `NLTagger(.nameType)` | `AppleNamedEntityRecognizer` | Apple Native (Throws `.unavailableOnPlatform` on Linux) |
| `VADER Sentiment` | N/A | `VADERSentimentAnalyzer` | Pure Swift (Static 200KB Lexicon, All Platforms) |
| OS ML Sentiment | `NLTagger(.sentimentScore)` | `NLSentimentAnalyzer` | Apple Native |
| `langid` | `NLLanguageRecognizer` | `AppleLanguageDetector` | Apple Native (Throws `.unavailableOnPlatform` on Linux) |
| `word2vec` / Vector Search | `NLEmbedding` | `AppleNLEmbedding` + `WordEmbeddings` | `WordEmbeddings` (All Platforms) / `AppleNLEmbedding` (Apple) |
| `NaiveBayesClassifier` | N/A | `MultinomialNaiveBayes`, `ComplementNaiveBayes` | Pure Swift (All Platforms) |
