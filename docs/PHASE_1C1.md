# Phase 1C.1 — Practical GGUF metadata export

Phase 1C.1 closes the inspection gap between parsing GGUF metadata and using that metadata to build Oracle's tokenizer and model manifest. The parser already retained typed values; this phase makes those values available through bounded terminal previews and complete machine-readable exports.

## Metadata-only mode

Use `--metadata` to omit tensor descriptors and payload telemetry:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf --metadata
mkdir -p model-reports
./build/oracle-gguf-inspect /path/to/model.gguf --metadata --json \
  > model-reports/model-metadata.json
```

`model-reports/` is ignored by Git so local model paths and large inspection outputs are not accidentally committed.

Text mode previews at most eight array items and truncates long strings. JSON mode emits complete values, including tokenizer vocabularies, merge tables, token types, and chat templates.

## Filtering

Exact-key and prefix filters imply metadata-only mode:

```bash
./build/oracle-gguf-inspect model.gguf \
  --metadata-key general.architecture --json

./build/oracle-gguf-inspect model.gguf \
  --metadata-prefix tokenizer.ggml. --json
```

`--metadata-key` and `--metadata-prefix` are mutually exclusive. Either filter may be combined with `--verify` when payload validation and metadata export must happen in the same process.

## JSON contracts

Full metadata reports use ordered entries so type information is preserved:

```json
{
  "path": "model.gguf",
  "version": 3,
  "metadata_count": 43,
  "selected_metadata_count": 2,
  "metadata": [
    {
      "key": "tokenizer.ggml.tokens",
      "type": "array",
      "element_type": "string",
      "length": 248320,
      "value": ["... complete values ..."]
    }
  ]
}
```

Normal `--json` and `--verify --json` reports use compact metadata entries. Scalar values remain present. Arrays retain their element type and length but omit their complete contents. Strings longer than 512 bytes retain a length and bounded preview.

This separation keeps routine model reports readable while preserving a lossless export path for tooling.

## Deliberate boundaries

Phase 1C.1 does not yet interpret architecture-specific keys, tokenize text, or execute MTP heads. Phase 1D will consume the exported values to create a validated tokenizer and model manifest without hard-coded IDs or dimensions.
