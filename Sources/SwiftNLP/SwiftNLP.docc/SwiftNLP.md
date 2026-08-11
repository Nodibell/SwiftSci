# ``SwiftNLP``

Natural Language Processing, Text Feature Extraction & Native Apple NaturalLanguage Integration.

## Overview

`SwiftNLP` provides high-performance text normalization, tokenization, linguistic tagging, sentiment analysis, entity recognition, language detection, word embeddings, text classification, and seamless `SwiftDataFrame` extensions.

### Key Capabilities

- **Tokenizers**: `AppleWordTokenizer` (Apple OS multi-lingual boundary detection), `SentenceTokenizer`, `RegexTokenizer`, `BPETokenizer`, and `NGramTokenizer`.
- **Linguistic Processing**: `PorterStemmer` (morphological suffix stripping), `POSTagger` (part-of-speech tagging), and `AppleLemmaTagger` (canonical base form lemmatization).
- **Named Entity Recognition**: `AppleNamedEntityRecognizer` extracting Person, Place, and Organization entities.
- **Sentiment Analysis**: `VADERSentimentAnalyzer` (valence score calculation with static 200KB lexicon) and `NLSentimentAnalyzer` (Apple OS ML sentiment scoring).
- **Language Detection & Embeddings**: `AppleLanguageDetector` (multi-lingual language recognition), `AppleNLEmbedding`, and `WordEmbeddings` (SIMD Accelerate vector dot-product optimization).
- **Text Classification**: `MultinomialNaiveBayes` and `ComplementNaiveBayes` classifiers for text classification tasks.
- **Ecosystem Pipelines**: `TextPipeline` and fluent `SwiftDataFrame` text extensions (`df.tokenizeColumn`, `df.stemColumn`, `df.analyzeSentiment`, `df.detectLanguage`, `df.extractEntities`).

### Example Usage

```swift
import SwiftNLP
import SwiftDataFrame

// Tokenization & Sentiment Analysis
let tokenizer = AppleWordTokenizer()
let tokens = tokenizer.tokenize(text: "SwiftSci 2.5.0 is fantastic!")

let vader = VADERSentimentAnalyzer()
let sentiment = vader.polarityScores(text: "SwiftSci is an amazing library!")
print(sentiment.compound) // > 0.5

// Fluent DataFrame Integration (supports both array literals and pre-typed [String] variables)
let articles: [String] = ["I love SwiftSci!", "This error is terrible."]
var df = try DataFrame(columns: [
    TypedColumn<String>(name: "text", values: articles)
])
let sentimentDF = try df.analyzeSentiment(column: "text")
```


## Topics

### Guides & Tutorials
- <doc:TextNormalizationAndTokenization>
- <doc:VectorizationAndEmbeddings>
