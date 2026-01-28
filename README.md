# Candle Qwen3-TTS

A Rust implementation of the Qwen3-TTS text-to-speech model using the [Candle](https://github.com/huggingface/candle) ML framework.

![qwen3-tts-rs logo](./qwen3-tts-rs.png)

## Features

- Complete implementation of the Qwen3-TTS architecture
- Speaker encoder (ECAPA-TDNN) for voice cloning
- 12Hz audio tokenizer (V2) for high-quality audio generation
- Three synthesis modes:
  - **CustomVoice**: Use predefined speaker voices
  - **VoiceDesign**: Create voices from natural language descriptions
  - **VoiceClone**: Clone voices from reference audio
- Batch processing for multiple texts
- Voice prompt caching for faster repeated generation
- URL-based audio loading for voice cloning
- Standalone tokenizer CLI for audio codec testing
- Full control over generation parameters
- Multi-language support: Chinese, English, Japanese, Korean, French, German, Spanish (+ auto-detect)

## Architecture Overview

Qwen3-TTS uses a hierarchical generation approach:

1. **Speaker Encoder**: Extracts speaker embeddings from reference audio using ECAPA-TDNN
2. **Talker Model**: Generates semantic tokens (codebook 0) using multimodal RoPE
3. **Code Predictor**: Generates acoustic tokens (codebooks 1-31)
4. **Audio Tokenizer**: Decodes all 32 codebooks to audio waveforms

## CLI Usage

### Basic Text-to-Speech

```bash
# Using a predefined speaker (CustomVoice mode)
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --output hello.wav

# With language specification
cargo run --release -- \
    --text "你好，世界！" \
    --speaker vivian \
    --language chinese \
    --output hello_chinese.wav
```

### Synthesis Modes

#### CustomVoice (Predefined Speakers)

Use built-in speaker voices with optional instructions:

```bash
cargo run --release -- \
    --text "Welcome to our service." \
    --speaker vivian \
    --instruct "Speak warmly and professionally" \
    --output welcome.wav
```

#### VoiceDesign (Natural Language Description)

Create a voice from a text description:

```bash
cargo run --release -- \
    --text "Hello, I'm your new assistant." \
    --voice-design "A warm, friendly female voice with a slight British accent" \
    --output designed_voice.wav
```

#### VoiceClone (Reference Audio)

Clone a voice from reference audio:

```bash
# X-vector only mode (faster, uses only speaker embedding)
# No --ref-text "..."
cargo run --release -- \
    --text "Quick voice cloning." \
    --ref-audio reference.wav \
    --output cloned_fast.wav

# From local file 
cargo run --release -- \
    --text "This is my cloned voice speaking." \
    --ref-audio reference.wav \
    --ref-text "The transcript of the reference audio." \
    --output output_cloned.wav

# From URL
cargo run --features cuda,flash-attn,audio-loading -- \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav" \
  --ref-text "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." \
  --text "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye." \
  --output output_cloned.wav
```

### Batch Processing

#### Using TXT

Create a text file with one text per line:

```bash
# inputs.txt
Hello, this is the first sentence.
This is the second sentence.
And here's the third one.
```

```bash
cargo run --release -- \
    --file inputs.txt \
    --speaker vivian \
    --output-dir ./outputs/
```

This generates `outputs/output_0.wav`, `outputs/output_1.wav`, etc.

#### Using JSON

For more control, use JSON format (detected automatically from `.json` extension):

```json
{
  "items": [
    {"text": "Hello in English!", "language": "english"},
    {"text": "你好！", "language": "chinese", "output": "chinese_greeting.wav"},
    {"text": "Another English sentence.", "language": "english"}
  ]
}
```

```bash
cargo run --release -- \
    --file inputs.json \
    --speaker vivian \
    --output-dir ./outputs/
```

### Voice Prompt Caching

Save computed voice prompts for reuse (avoids recomputing speaker embeddings):

```bash
# Save voice prompt while generating
cargo run --release -- \
    --text "First generation." \
    --ref-audio reference.wav \
    --ref-text "Reference transcript." \
    --save-prompt voice_prompt.safetensors \
    --output first.wav

# Reuse saved prompt (faster, no need for reference audio)
cargo run --release -- \
    --text "Second generation with same voice." \
    --load-prompt voice_prompt.safetensors \
    --output second.wav

# Create prompt without generating audio
cargo run --release -- \
    --ref-audio reference.wav \
    --ref-text "Reference transcript." \
    --save-prompt voice_prompt.safetensors
```

### Generation Parameters

#### Talker Parameters (Semantic Token Generation)

```bash
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --temperature 0.9 \
    --top-k 50 \
    --top-p 1.0 \
    --repetition-penalty 1.05 \
    --max-tokens 2048 \
    --output output.wav

# Greedy decoding (deterministic)
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --greedy \
    --output output.wav

# Set random seed for reproducibility
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --seed 42 \
    --output output.wav
```

##### Max Tokens (default: 2048)

If you want to generate long form text you will need to adjust the `--max-tokens`.

The "Hz" in the model names literally means "tokens per second".

- v1 25Hz = 25 tokens/second = 40ms per token
- v2 12Hz = 12.5 tokens/second = 80ms per token

Given: `tokens = duration_seconds × token_rate_hz`

| max_tokens | 12Hz(v2)| 25Hz(v1)|
|------------|-------- |---------|
| 2,000      | 2m 40s  | 1m 20s  |
| 4,000      | 5m 20s  | 2m 40s  |
| 8,000      | 10m     | 5m 20s  |
| 16,000     | 21m     | 10m     |
| 32,000     | 42m     | 20m     |

##### Per Page

**Reading time for 12 pages**
- Average: 500 words per page
- Speech rate: 150 words per minute (conversational pace)

| Pages | Words  | Duration | 12Hz    | 25Hz    |
|-------|------- |----------|---------|---------|
| 1     | 500    | 3.3 min  | 2,500   | 5,000   |
| 5     | 2,500  | 17 min   | 12,750  | 25,500  |
| 12    | 6,000  | 40 min   | 30,000  | 60,000  |
| 25    | 12,500 | 83 min   | 62,250  | 124,500 |
```

#### Subtalker Parameters (Acoustic Token Generation)

Control the code predictor that generates codebooks 1-31:

```bash
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --subtalker-temperature 0.9 \
    --subtalker-top-k 50 \
    --subtalker-top-p 1.0 \
    --output output.wav

# Disable subtalker sampling (greedy)
cargo run --release -- \
    --text "Hello, world!" \
    --speaker vivian \
    --no-subtalker-sample \
    --output output.wav
```

### Hardware Options

```bash
# Use CPU
cargo run --release -- \
    --text "Hello!" \
    --speaker vivian \
    --device cpu \
    --output output.wav

# Use CUDA GPU
cargo run --release -- \
    --text "Hello!" \
    --speaker vivian \
    --device cuda \
    --output output.wav

# Use Metal (macOS)
cargo run --release -- \
    --text "Hello!" \
    --speaker vivian \
    --device metal \
    --output output.wav

# Set data type
cargo run --release -- \
    --text "Hello!" \
    --speaker vivian \
    --dtype bf16 \
    --output output.wav
```

### Tokenizer CLI

Standalone CLI for audio encoding/decoding (codec testing):

```bash
# Encode audio to codes
cargo run --release --example tokenizer_cli --features audio-loading -- encode \
    --input audio.wav \
    --output codes.json

# Decode codes back to audio
cargo run --release --example tokenizer_cli --features audio-loading -- decode \
    --input codes.json \
    --output reconstructed.wav

# Round-trip test (encode then decode)
cargo run --release --example tokenizer_cli --features audio-loading -- roundtrip \
    --input audio.wav \
    --output reconstructed.wav
```

The codes JSON format:

```json
{
  "sample_rate": 24000,
  "num_codebooks": 32,
  "codes": [[1995, 1642, ...], ...]
}
```