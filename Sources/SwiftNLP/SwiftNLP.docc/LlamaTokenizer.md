# Llama checkpoint tokenization

`BPETokenizer(llama3TokenizerJSON:)` loads the byte-level BPE pipeline used by the pinned Llama 3.2 1B Instruct checkpoint.

## Construction

```swift
import Foundation
import SwiftNLP

let data = try Data(contentsOf: URL(fileURLWithPath: "/path/to/checkpoint/tokenizer.json"))
let tokenizer = try BPETokenizer(llama3TokenizerJSON: data)
let ids = tokenizer.encode(text: "Hello world!")
// [9906, 1917, 0] for the pinned checkpoint
```

The initializer accepts the Llama 3 regular-expression split, byte alphabet, vocabulary-first BPE, special-token definitions, and single-sequence beginning-of-text template. Merge rules can be strings or two-element string arrays.

The initializer rejects unsupported normalization, padding, truncation, token-stripping rules, and post-processing. Invalid byte vocabularies, duplicate IDs, and invalid merge rules also throw `BPETokenizerConfigurationError`. Malformed JSON throws a Foundation decoding error. This API is not a general Hugging Face tokenizer interpreter.

## Encoding and decoding

`encode(text:)` returns the token IDs for exactly the supplied text. It preserves whitespace and recognizes literal special-token strings such as `<|eot_id|>`. Special-token strings in untrusted content therefore act as control tokens; this method does not escape them.

`encode(text:addSpecialTokens: true)` prepends the configured beginning-of-text token. It applies that template even when the input already starts with a literal beginning-of-text token, matching the checkpoint's post-processor.

`decode(tokens:)` preserves whitespace and includes special-token text. Unknown IDs are omitted, consistent with the existing tokenizer API. Incomplete UTF-8 sequences decode with replacement characters, so decoding individual generated tokens can differ from decoding the complete token sequence.

## Chat formatting

The initializer reads `tokenizer.json`. It does not execute the chat template in `tokenizer_config.json`.

An already rendered checkpoint chat prompt uses `encode(text:)`, because that prompt contains its own beginning-of-text token. SwiftSci's generic `ChatTemplate.llama3` renderer is separate from the checkpoint template. Its output has not been established as equivalent to the checkpoint's system headers, dates, or tool formatting.

## Compatibility and validation

The existing `BPETokenizer(vocab:merges:)` initializer retains its word-ending BPE behavior, including `</w>` and whitespace normalization. Callers select checkpoint behavior through the new initializer.

Self-contained `LlamaTokenizerTests` cover byte preservation, Unicode, special tokens, merge order, digit boundaries, and invalid configurations without model downloads. `LlamaTokenizerCheckpointTests` is opt-in through `SWIFTSCI_LLAMA_TOKENIZER_JSON`, which names the local checkpoint's `tokenizer.json`. That suite verifies the file hash before checking reference IDs and exact decoded bytes.

The pinned tokenizer comes from `mlx-community/Llama-3.2-1B-Instruct-bf16`, revision `863c846a9ac6fad4e49e1743d52984dff262e953`. Its SHA-256 is `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`. Reference IDs were produced with `tokenizers` 0.23.2. The rendered-chat cases use `transformers` 5.17.0 and a fixed date. Checkpoint files remain external to the repository.

Tokenizer parity alone does not establish native checkpoint loading or generation correctness.
