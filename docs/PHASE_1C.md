# Phase 1C — Memory-mapped GGUF tensor access

Phase 1C turns GGUF tensor descriptors into validated, read-only views over real model-file bytes. It deliberately stops before decoding or executing weights.

## Memory-mapped files

`oracle::core::MappedFile` owns a read-only file mapping with RAII cleanup and move semantics. Linux uses `mmap` with `MAP_PRIVATE`; the portable fallback reads the file into owned memory so the public interface remains stable.

A mapping exposes a `std::span<const std::byte>` and never hands out writable model storage.

## GGML type layouts

`oracle::model::GgmlTypeLayout` records the storage contract needed to validate a tensor:

- GGML type identifier and name
- elements represented by one storage block
- bytes occupied by one storage block
- whether the type is quantized

Removed and unknown type identifiers are rejected. Tensor byte size is computed by validating the first GGML dimension as a row, then multiplying by the remaining dimensions with overflow checks.

## Validated tensor views

`oracle::model::MappedGgufModel` combines the Phase 1B metadata parser with a file mapping. Construction validates:

- the parsed file size matches the mapped file size
- every GGML type has a known layout
- quantized row dimensions are divisible by their block size
- absolute tensor offsets do not overflow
- complete payloads fit inside the mapped file
- tensor payloads do not overlap
- tensor names remain unique

Each `GgufTensorView` points directly into the mapped model file. No tensor payload is copied or decoded.

## Inspection

Metadata-only inspection remains available:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf
```

Map and validate every tensor payload:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf --verify
./build/oracle-gguf-inspect /path/to/model.gguf --verify --json
```

Inspect one named tensor:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf --tensor token_embd.weight
./build/oracle-gguf-inspect /path/to/model.gguf --tensor token_embd.weight --json
```

The verified output includes mapped bytes, validated payload bytes, tensor span, absolute offsets, element counts, and storage-block information.

## Deliberate boundaries

Phase 1C does not yet provide:

- F16, BF16, or quantized weight decoding
- architecture-specific tensor naming rules
- tokenization
- transformer attention or KV-cache execution
- model generation

Those layers will consume the validated views without changing the GGUF parser or file-mapping contract.
