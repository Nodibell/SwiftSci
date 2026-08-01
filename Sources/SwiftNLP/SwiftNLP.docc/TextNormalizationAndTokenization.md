# Text Normalization, Tokenization & Linguistic Tagging

Clean text, strip punctuation, tokenize sentences/words, tag parts of speech, extract entities, stem and lemmatize tokens, and score sentiment using NLTK-equivalent algorithms and Apple's native `NaturalLanguage` framework.

## Overview

Prepare raw unstructured text for machine learning pipelines with high performance and zero external dependencies.

### 1. Multi-lingual Word & Sentence Tokenization

```swift
import SwiftNLP

// Apple OS multi-lingual word boundary detection
let wordTokenizer = AppleWordTokenizer()
let words = wordTokenizer.tokenize(text: "SwiftSci 2.5.0 is fantastic for NLP!")

// Sentence boundary tokenizer
let sentenceTokenizer = SentenceTokenizer()
let sentences = sentenceTokenizer.tokenize(text: "First sentence. Second sentence!")
```

### 2. POS Tagging, Lemmatization & Entity Extraction

```swift
// Part-of-speech tagging
let posTagger = POSTagger()
let tags = posTagger.tag(text: "SwiftSci runs fast")
// [TaggedToken(token: "SwiftSci", tag: .noun), TaggedToken(token: "runs", tag: .verb), ...]

// Lemmatization (base canonical form)
let lemmaTagger = AppleLemmaTagger()
let lemmas = lemmaTagger.lemmatize(text: "running faster")
// ["run", "fast"]

// Named Entity Recognition
let ner = AppleNamedEntityRecognizer()
let entities = ner.extractEntities(text: "Tim Cook visited Apple in Cupertino.")
```

### 3. Sentiment Analysis (VADER & NLSentiment)

```swift
// Pure Swift VADER Sentiment Analysis
let vader = VADERSentimentAnalyzer()
let score = vader.polarityScores(text: "SwiftSci is an amazing library!")
print(score.compound) // > 0.5 (positive)

// Apple OS ML Sentiment Analyzer
let nlSentiment = NLSentimentAnalyzer()
let sentimentScore = nlSentiment.score(text: "I love coding in Swift!")
```
