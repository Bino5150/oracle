# Phase 1D — Qwen3.5 manifest, tokenizer, and chat formatting

Phase 1D moves Oracle from structural GGUF inspection to deterministic model-facing text handling. It adds a validated Qwen3.5 model manifest, a dependency-free GPT-2 byte-level BPE tokenizer using the Qwen3.5 pre-tokenization rules, and a native formatter for the chat/tool schema embedded in the target GGUF models.

## Model manifest

`Qwen35Manifest` extracts and validates:

- total, backbone, and optional NextN/MTP block counts
- context, embedding, and feed-forward dimensions
- attention and KV-head counts
- key and value widths
- RMSNorm epsilon
- RoPE base, dimension count, and section layout
- SSM convolution, state, group, rank, and inner dimensions
- full-attention interval
- vocabulary size and optional BOS, EOS, and padding IDs

MTP is resolved as an appended prediction head:

```text
backbone_block_count = qwen35.block_count - qwen35.nextn_predict_layers
```

For the validated target models this resolves both the base and MTP variants to the same 32-block backbone. When tensor descriptors are present, the loader validates the embedding dimensions and requires appended `nextn` tensors for MTP metadata.

## Tokenizer

`Qwen35Tokenizer` loads:

- `tokenizer.ggml.tokens`
- `tokenizer.ggml.token_type`
- `tokenizer.ggml.merges`
- optional BOS, EOS, and padding token IDs
- the embedded chat-template string

The implementation provides:

- GPT-2 byte-to-Unicode mapping and reverse decoding
- Qwen3.5 Unicode-aware pre-tokenization
- deterministic rank-ordered BPE merging
- byte fallback for arbitrary UTF-8 input
- explicit special-token parsing mode
- optional special-token omission during decoding
- multilingual, combining-mark, emoji, whitespace, and raw-byte-safe round trips

Unicode letter, mark, number, and whitespace category ranges are compiled into Oracle, avoiding locale-dependent behavior or an external Unicode library. The generated tables use Unicode 15.1 data.

Normal encoding does not interpret control-token text. Callers must explicitly enable special-token parsing for formatted prompts. This prevents user text such as `<|im_start|>` from silently becoming a control token.

## Chat formatting

`format_qwen35_chat` implements the behavior of the Qwen3.5 template carried by the two target GGUF files:

- system, user, assistant, and tool roles
- optional tool-definition preamble
- assistant reasoning in `<think>` blocks
- OpenAI-style function calls in `<tool_call>` blocks
- adjacent tool results grouped into a user turn
- generation prompts ending with the assistant reasoning prefix

The formatter is a native, typed implementation of this schema. It is not a general-purpose Jinja interpreter.

## CLI

Inspect a manifest:

```bash
./build/oracle-tokenize model.gguf --manifest
./build/oracle-tokenize model.gguf --manifest --json
```

Encode and decode:

```bash
./build/oracle-tokenize model.gguf --encode "Hello Oracle" --json
./build/oracle-tokenize model.gguf --encode "<|im_start|>assistant" \
  --parse-special --json
./build/oracle-tokenize model.gguf --decode "248045,74455" --json
```

Format and tokenize a simple chat:

```bash
./build/oracle-tokenize model.gguf \
  --chat-system "You are Lumina." \
  --chat-user "Hello Oracle." \
  --tokenize-chat --json
```

The public C++ chat API supports tool definitions, tool calls, tool responses, and reasoning content beyond the compact CLI fixture flags.

## Validation

Phase 1D includes tests for:

- base and MTP manifest extraction
- 33-total/32-backbone MTP separation
- tensor and metadata mismatch rejection
- GPT-2 byte vocabulary validation
- deterministic BPE merges
- multilingual and combining-mark round trips
- one-digit Qwen3.5 pre-tokenization behavior
- explicit control-token parsing
- special-token skipping during decode
- tool-aware chat formatting and prompt tokenization

The complete suite is validated with GCC, Clang, AddressSanitizer, UndefinedBehaviorSanitizer, and leak detection. The full 248,320-token metadata exports from the target base and MTP models were also converted into metadata-only GGUF fixtures and loaded by `oracle-tokenize`. Token IDs for representative English, whitespace, punctuation, numeric, Chinese, Korean, Arabic, Devanagari, emoji, and combining-mark samples were compared against an independent implementation of the published Qwen3.5 regex and GPT-2 BPE procedure.

## Deliberate limits

Phase 1D does not:

- decode F16, BF16, Q5_K, or Q6_K tensor payloads
- perform embedding lookup or model execution
- implement RoPE, SSM, attention, KV cache, or sampling
- evaluate arbitrary Jinja chat templates
- expose an HTTP server

Those execution primitives begin in Phase 1E.
