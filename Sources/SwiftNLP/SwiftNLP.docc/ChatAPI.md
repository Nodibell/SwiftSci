# Chat Templates & Multi-Turn Conversations

Format structured multi-turn dialogue into model-specific prompts for Llama-3, ChatML, and Mistral architectures.

## Overview

Modern causal language models require specialized prompt formatting with model-specific control tokens to delineate system instructions, user inputs, and assistant turns. `SwiftNLP` provides structured `ChatMessage` and `ChatTemplate` abstractions that render conversation histories into raw text or encoded token sequences without modifying the core `Tokenizer` interface.

## 1. Defining Chat Messages

A conversation is represented as an array of `ChatMessage` value types:

```swift
import SwiftNLP

let conversation: [ChatMessage] = [
    ChatMessage(role: .system, content: "You are a helpful scientific data analysis assistant."),
    ChatMessage(role: .user, content: "What is the difference between RoPE and learned positional embeddings?"),
    ChatMessage(role: .assistant, content: "Rotary Position Embeddings (RoPE) encode relative position via complex plane rotations..."),
    ChatMessage(role: .user, content: "Can you provide a code sample?")
]
```

## 2. Supported Chat Templates

`ChatTemplate` provides presets matching standard foundation models:

| Preset | Target Models | Format Structure |
|---|---|---|
| **`.llama3`** | Llama 3, Llama 3.1, Llama 3.2 | `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` |
| **`.chatML`** | Qwen 2.5, Hermes, Yi | `<|im_start|>role\ncontent<|im_end|>` |
| **`.mistral`** | Mistral 7B, Mixtral | `[INST] system\n\nuser [/INST] assistant` |

## 3. Rendering & Tokenization

### Formatting to Prompt String

```swift
let template = ChatTemplate.llama3
let formattedPrompt = template.render(messages: conversation, addGenerationPrompt: true)
print(formattedPrompt)
// Appends `<|start_header_id|>assistant<|end_header_id|>\n\n` to prime the model for completion
```

### Direct Tokenizer Encoding

Use the high-level `encode` convenience to tokenize rendered chats directly into token IDs:

```swift
let tokenizer = BPETokenizer(...)
let tokenIDs = try template.encode(
    messages: conversation,
    using: tokenizer,
    addGenerationPrompt: true
)
```

## Topics

### Chat Types
- ``ChatMessage``
- ``ChatRole``
- ``ChatTemplate``
