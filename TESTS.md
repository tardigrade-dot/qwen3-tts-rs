# Testing Guide

## Quick Start

```bash
# CPU only
cargo test --features test-all

# With CUDA (Linux/Windows with NVIDIA GPU)
cargo test --features test-all-cuda -- --test-threads=1

# With Metal (macOS)
cargo test --features test-all-macos -- --test-threads=1
```

**Note:** First run downloads ~1-2GB of model weights from HuggingFace.

**Important:** GPU tests must run with `--test-threads=1` to avoid out-of-memory errors from parallel model loading.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `integration-tests` | Enables integration tests (model downloads) |
| `test-all` | Integration tests, CPU only |
| `test-all-cuda` | Integration tests + CUDA support |
| `test-all-macos` | Integration tests + Metal support |

GPU support requires compile-time features because candle links against CUDA/Metal libraries. There's no runtime auto-detection.

## Test Categories

### Unit Tests (no features required)

```bash
cargo test -p qwen_tts
```

Tests struct creation, validation, config parsing. No model downloads.

### Integration Tests (require `integration-tests` feature)

Located in `tests/integration_tests.rs`:

| Module | Tests |
|--------|-------|
| `loader_tests` | Model loading, config parsing |
| `text_processor_tests` | Tokenization, unicode, edge cases |
| `tokenizer_tests` | Audio tokenizer encode/decode |
| `generation_tests` | Speech generation (CustomVoice, VoiceDesign, VoiceClone) |
| `error_path_tests` | Invalid inputs, wrong model types |

### Slow Tests

Some tests are marked `#[ignore]` because they require full model loading with audio tokenizer:

```bash
# Run ignored tests
cargo test --features test-all-cuda -- --ignored

# Run all tests including ignored
cargo test --features test-all-cuda -- --test-threads=1 --include-ignored
```

## Troubleshooting

**CUDA_ERROR_OUT_OF_MEMORY:**
Tests run in parallel by default. Use `--test-threads=1` to run sequentially:
```bash
cargo test --features test-all-cuda -- --test-threads=1
```