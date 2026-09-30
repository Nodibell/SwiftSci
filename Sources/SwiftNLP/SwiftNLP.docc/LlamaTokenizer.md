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

## Streaming generated text

Create a separate decoder for each generated response:

```swift
var stream = tokenizer.makeStreamDecoder()
for token in generatedTokenIDs {
    if let text = stream.append(token) {
        output(text)
    }
}
output(stream.finish())
```

For Llama byte-level tokenizers, `append` retains at most three trailing bytes of an
incomplete UTF-8 scalar. It returns `nil` when no complete text is available yet.
Complete text is emitted immediately. `finish` flushes unfinished bytes using the
same replacement-character rules as `decode(tokens:)`; repeated calls return an
empty string. Decoding does not normalize whitespace or discard literal replacement
characters. Each token is processed once without decoding the entire response again.

Both `TransformerDecoder` generation APIs use this decoder. Other tokenizers retain
per-token decoding through the default `Tokenizer.makeStreamDecoder()` implementation.
A custom byte tokenizer can supply `TokenStreamDecoder(decodeBytes:)` to opt in.

## Chat formatting

The initializer reads `tokenizer.json`. It does not execute the chat template in `tokenizer_config.json`.

`ChatTemplate.llama32Instruct(date:)` implements the pinned checkpoint's text-conversation formatting. The date is explicit, so the same inputs produce the same prompt without consulting the clock or locale.

```swift
let template = ChatTemplate.llama32Instruct(date: "30 Sep 2026")
let ids = template.encode(
	messages: [.system("Be concise."), .user("What is 2 + 2?")],
	using: tokenizer
)
```

The renderer emits BOS once, followed by the dated system header. It trims ordinary message content using the checkpoint's whitespace rules. It emits string tool responses as JSON strings under the `ipython` role, preserving their whitespace. A leading system message supplies the system content. Later system messages retain their position in the conversation.

The renderer supports the checkpoint's `tools=None` path. It does not accept tool definitions, structured tool calls, structured tool results, or multimodal content. As a Swift API convenience, an empty conversation produces the dated system header and an optional assistant header. The reference's conversation wrapper requires a nonempty message list, so this empty-input behavior is tested separately.

An already rendered prompt uses `encode(text:)`, because it contains its own BOS. `ChatTemplate.encode` also uses this method and does not add another BOS. The generic `ChatTemplate.llama3` style retains its existing behavior and is separate from this checkpoint-specific style.

## Compatibility and validation

The existing `BPETokenizer(vocab:merges:)` initializer retains its word-ending BPE behavior, including `</w>` and whitespace normalization. Callers select checkpoint behavior through the new initializer.

Self-contained `LlamaTokenizerTests` cover byte preservation, Unicode, special tokens, merge order, digit boundaries, and invalid configurations without model downloads. `LlamaTokenizerCheckpointTests` is opt-in through `SWIFTSCI_LLAMA_TOKENIZER_JSON`, which names the local checkpoint's `tokenizer.json`. That suite verifies the file hash before checking reference IDs and exact decoded bytes.

The pinned tokenizer comes from `mlx-community/Llama-3.2-1B-Instruct-bf16`, revision `863c846a9ac6fad4e49e1743d52984dff262e953`. Its SHA-256 is `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`. Reference IDs were produced with `tokenizers` 0.23.2. The rendered-chat cases use `transformers` 5.17.0 and a fixed date. Checkpoint files remain external to the repository.

`Llama32ChatTemplateTests` checks rendered bytes against the pinned checkpoint template, including ordinary messages, string tool responses, Unicode, and generation-header control. With `SWIFTSCI_LLAMA_TOKENIZER_JSON` set, it also checks complete prompt token IDs. The tokenizer configuration SHA-256 is `9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e` at the same pinned revision.

Tokenizer and chat-template parity do not establish native checkpoint loading or generation correctness.
