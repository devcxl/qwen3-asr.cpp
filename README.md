# Qwen3-ASR.cpp

A high-performance C++ implementation of Qwen3-ASR and Qwen3-ForcedAligner using the GGML tensor library. Optimized for Apple Silicon with Metal GPU acceleration, providing fast speech recognition and word-level timestamp alignment.

## Features

- **Automatic Speech Recognition (ASR)**: Transcribe audio files to text in 30+ languages
- **Forced Alignment**: Align reference text to audio with word-level timestamps
- **Combined Pipeline** (`--transcribe-align`): Automatically runs ASR then alignment with auto language detection
- **Flash Attention**: Uses `ggml_flash_attn_ext()` for fast decoding (3.7x speedup)
- **Metal GPU Acceleration**: Optimized for Apple Silicon with dual CPU+Metal backend
- **Accelerate/vDSP**: Highly optimized mel spectrogram computation (45x speedup)
- **mmap Weight Loading**: Zero-copy GPU transfer for fast model initialization
- **F16 KV Cache**: Reduced memory bandwidth with half-precision key-value cache
- **Korean Word Splitting**: Soynlp LTokenizer algorithm with 18K-word dictionary
- **Quantization Support**: Q8_0 quantization for reduced memory usage (~40% smaller)
- **Pure C++17**: No Python runtime required for inference

## Supported Models

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-asr-0.6b-f16.gguf` | ~1.8 GB | ASR model, F16 precision |
| `qwen3-asr-0.6b-q8_0.gguf` | ~1.3 GB | ASR model, Q8_0 quantized |
| `qwen3-forced-aligner-0.6b-f16.gguf` | ~1.8 GB | Forced alignment model |

## Requirements

- CMake 3.14+
- C++17 compatible compiler (Clang 7+, GCC 8+, MSVC 2019+)
- Apple Silicon recommended (Metal GPU support)
- GGML library (included as submodule)

## Building

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/predict-woo/qwen3-asr.cpp.git
cd qwen3-asr.cpp

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

On Linux, replace `$(sysctl -n hw.ncpu)` with `$(nproc)`.

### 5. Python Bindings

Use the ASR engine directly from Python via pybind11 bindings:

```bash
# Install with pip (editable, requires CMake + C++17 compiler)
pip install -e . --no-build-isolation

# Or build with CMake directly
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DQWEN3_ASR_BUILD_PYTHON=ON
cmake --build . -j$(nproc)
```

**Python usage:**

```python
from qwen3_asr import Qwen3ASR, TranscribeParams

# Load model
asr = Qwen3ASR()
asr.load_model("models/qwen3-asr-0.6b-f16.gguf")

# Transcribe audio file
result = asr.transcribe("audio.wav")
print(result.text)  # "language Chinese ..."

# With custom parameters
params = TranscribeParams()
params.max_tokens = 2048
params.n_threads = 8
result = asr.transcribe("audio.wav", params)
print(f"Decoded in {result.t_total_ms:.0f}ms")

# Transcribe raw samples (numpy array)
import numpy as np
samples, sr = load_audio_file("audio.wav")  # (ok, samples, sample_rate)
result = asr.transcribe_raw(np.array(samples, dtype=np.float32))
print(result.text)

# Context manager — auto cleanup
with Qwen3ASR() as asr:
    asr.load_model("models/qwen3-asr-0.6b-f16.gguf")
    result = asr.transcribe("audio.wav")
    print(result.text)
```

The Python module wraps the C++ `Qwen3ASR` class with RAII semantics — destructors run automatically on garbage collection, so there are no memory leaks.

See [python/qwen3_asr/](python/qwen3_asr/) for the module sources.

### 4. HTTP Server (OpenAI-compatible API)

Start the ASR HTTP server — provides OpenAI-compatible endpoints for audio transcription, supporting concurrent requests with slot-based parallelism:

```bash
# Start server (default: 127.0.0.1:8080, 1 parallel slot)
./build/qwen3-asr-server -m models/qwen3-asr-0.6b-f16.gguf

# With 4 parallel inference slots, bind to all interfaces
./build/qwen3-asr-server -m models/qwen3-asr-0.6b-f16.gguf -np 4 -t 4 --host 0.0.0.0 --port 8080
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (shorter alias) |
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio (OpenAI-compatible) |

**curl examples (all response formats):**

```bash
# JSON response: {"text": "... transcription ..."}
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=json"

# Plain text response
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"

# SubRip (SRT) subtitle format
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=srt"

# WebVTT subtitle format
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=vtt"

# Verbose JSON with duration info
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"

# Health check
curl http://localhost:8080/v1/health
```

**Python client (OpenAI SDK compatible):**

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8080/v1',
    api_key='not-needed'  # no auth required
)

with open('audio.wav', 'rb') as f:
    transcript = client.audio.transcriptions.create(
        model='qwen3-asr',
        file=f
    )
    print(transcript.text)
```

**Server options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | (required) | Path to Qwen3 ASR GGUF model |
| `--host` | `127.0.0.1` | Host address to bind |
| `--port` | `8080` | Port to listen on |
| `-np, --n-parallel` | `1` | Number of parallel inference slots |
| `-t, --threads` | `4` | CPU threads for inference |
| `--debug` | off | Enable debug output |

**Server architecture:**

The server is built on a **slot-based parallelism model**:

```
┌─────────────────────────────────────────────────┐
│                  HTTP Server (httplib)           │
│                                                  │
│  POST /v1/audio/transcriptions                   │
│         │                                        │
│         ▼                                        │
│  ┌──────────────────────────────┐                │
│  │     Slot Pool (mutex-protected)               │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ │                │
│  │  │Slot 0│ │Slot 1│ │Slot 2│ │ ... n_parallel │
│  │  └──────┘ └──────┘ └──────┘ │                │
│  └──────────────────────────────┘                │
│         │  │  │                                  │
│         ▼  ▼  ▼                                  │
│  ┌──────────────────────────────┐                │
│  │   Shared Qwen3ASR Model      │  (read-only)   │
│  └──────────────────────────────┘                │
└─────────────────────────────────────────────────┘
```

1. **Slot Pool (`asr_slot_pool`)**: A fixed-size pool of pre-allocated inference slots, protected by a mutex. Each slot holds its own input/output buffers.
2. **Acquire → Infer → Release**: On each request, the server acquires a free slot from the pool, runs inference synchronously on that slot, then releases it. If all slots are busy, the request gets a `429 Too Many Requests` response.
3. **Shared model**: The underlying `Qwen3ASR` model instance is shared across all slots. The model itself is read-only during inference (GGML graph execution), so no locking is needed at the model level.
4. **WAV parsing in-memory**: Audio files are parsed directly from the HTTP request body — no temp files. Supports PCM 16-bit, 32-bit float, 32-bit int, and 8-bit WAV formats, with automatic mono downmix and 16kHz resampling.
5. **No external JSON dependency**: JSON responses are built with lightweight string helpers — no nlohmann/json or other dependency.
6. **Timeout & limits**: 120s read/write timeout, 50MB max payload size.

The `-np` flag controls concurrency: each parallel slot uses independent audio/text buffers, so requests don't interfere. The recommended value depends on your CPU cores — start with `-np 4` on a 4+ core machine.

## Quick Start

### 1. Transcription (ASR)

Transcribe audio files to text:

```bash
# Basic transcription
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav

# With quantized model (faster, less memory)
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-q8_0.gguf -f audio.wav

# Save output to file
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav -o transcript.txt

# Multi-threaded processing
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f audio.wav -t 8
```

### 2. Forced Alignment

Align reference text to audio with word-level timestamps:

```bash
# Basic alignment
./build/qwen3-asr-cli \
  -m models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --align \
  --text "transcript text" \
  --lang korean

# Save alignment to JSON file
./build/qwen3-asr-cli \
  -m models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --align \
  --text "Hello world" \
  -o alignment.json
```

### 3. Combined Pipeline (Transcribe + Align)

Automatically transcribe and then align the result (recommended):

```bash
./build/qwen3-asr-cli \
  -m models/qwen3-asr-0.6b-f16.gguf \
  --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf \
  -f audio.wav \
  --transcribe-align
```

This mode automatically:
- Runs ASR to get the transcript
- Detects the language from the ASR output
- Runs forced alignment with the detected language
- Outputs word-level timestamps as JSON

### Output Formats

**Transcription** outputs plain text:
```
language Korean 안녕하세요 여러분 오늘은...
```

**Forced Alignment** outputs JSON with word-level timestamps:
```json
{
  "words": [
    {"word": "안녕하세요", "start": 0.000, "end": 0.480},
    {"word": "여러분", "start": 0.480, "end": 0.880},
    {"word": "오늘은", "start": 0.880, "end": 1.200}
  ]
}
```

## Performance

Benchmark on 92-second Korean audio, Apple M2 Pro (10-core CPU, 16-core GPU):

| Stage | Time |
|-------|------|
| Mel spectrogram | 98 ms |
| Audio encoding | 715 ms |
| Text decoding (323 tokens) | 4,194 ms |
| **ASR Total** | **5,007 ms** |
| Forced alignment (183 words) | 12,998 ms |
| **Combined Total** | **18,005 ms** |

**Memory Usage:** ~247 MB RSS, ~294 MB Metal

### Key Optimizations

- **Flash Attention** (`ggml_flash_attn_ext`): 3.7x decode speedup vs. standard attention
- **Metal GPU Dual Backend**: Automatic scheduling between CPU and GPU for optimal performance
- **mmap + Zero-Copy GPU Transfer**: Fast model loading via `ggml_backend_dev_buffer_from_host_ptr`
- **F16 KV Cache**: Half-precision key-value cache reduces memory bandwidth
- **Selective Logits**: Only compute last token logits for lm_head (saves computation)
- **Weight Tying**: token_embd = output weight (saves memory)
- **vDSP/Accelerate Mel**: 45x speedup for mel spectrogram computation on Apple platforms
- **Korean Word Splitting**: Soynlp LTokenizer port with bundled 18K-word dictionary

## Model Conversion

Convert HuggingFace models to GGUF format:

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Convert ASR model (F16)
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ASR-0.6B \
    --output models/qwen3-asr-0.6b-f16.gguf \
    --type f16

# Convert ASR model (Q8_0 quantized)
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ASR-0.6B \
    --output models/qwen3-asr-0.6b-q8_0.gguf \
    --type q8_0

# Convert ForcedAligner model
python scripts/convert_hf_to_gguf.py \
    --input /path/to/Qwen3-ForcedAligner-0.6B \
    --output models/qwen3-forced-aligner-0.6b-f16.gguf \
    --type f16
```

## Supported Languages

The model supports 30+ languages:

| Language | Code | Language | Code |
|----------|------|----------|------|
| Chinese (Mandarin) | zh | English | en |
| Cantonese | yue | Japanese | ja |
| Korean | ko | German | de |
| French | fr | Spanish | es |
| Italian | it | Portuguese | pt |
| Russian | ru | Arabic | ar |
| Hindi | hi | Thai | th |
| Vietnamese | vi | Indonesian | id |
| Malay | ms | Turkish | tr |
| Polish | pl | Dutch | nl |
| Swedish | sv | Norwegian | no |
| Danish | da | Finnish | fi |
| Greek | el | Czech | cs |
| Hungarian | hu | Romanian | ro |
| Ukrainian | uk | Hebrew | he |

## Audio Requirements

- **Format**: WAV (PCM)
- **Sample rate**: 16 kHz
- **Channels**: Mono
- **Bit depth**: 16-bit

Convert audio with ffmpeg:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## Performance Profiling

Build with timing instrumentation to see detailed breakdowns:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DQWEN3_ASR_TIMING=ON
cmake --build . -j$(sysctl -n hw.ncpu)

# Run with --profile flag
./qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f sample.wav --profile
```

For production builds, omit `-DQWEN3_ASR_TIMING=ON` to remove timing overhead.

## Project Structure

```
qwen3-asr.cpp/
├── src/
│   ├── main.cpp              # CLI entry point
│   ├── server.cpp/h          # HTTP server (OpenAI-compatible API)
│   ├── qwen3_asr.cpp/h       # High-level ASR API
│   ├── forced_aligner.cpp/h  # Forced alignment implementation
│   ├── audio_encoder.cpp/h   # Audio feature encoder
│   ├── text_decoder.cpp/h    # Text decoder (Qwen2 architecture)
│   ├── mel_spectrogram.cpp/h # Mel spectrogram computation
│   ├── audio_injection.cpp/h # Audio-text embedding injection
│   ├── gguf_loader.cpp/h     # GGUF model loading
│   └── timing.h              # Timing instrumentation macros
├── python/
│   ├── bindings.cpp          # pybind11 bindings (_core module)
│   └── qwen3_asr/
│       ├── __init__.py       # Python package init
│       └── _core*.so         # Compiled extension (gitignored)
├── cmake/
│   └── python.cmake          # CMake config for Python bindings
├── pyproject.toml             # pip build config (scikit-build-core)
├── tests/
│   ├── test_mel.cpp          # Mel spectrogram tests
│   ├── test_encoder.cpp      # Audio encoder tests
│   ├── test_decoder.cpp      # Text decoder tests
│   └── reference/            # Reference data for validation
├── scripts/
│   └── convert_hf_to_gguf.py # Model conversion script
├── assets/
│   └── korean_dict_jieba.dict # Korean word dictionary (17,968 words)
├── models/                   # GGUF model files (not tracked in git)
├── ggml/                     # GGML library (git submodule)
└── CMakeLists.txt
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) - Tensor library for machine learning
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) - Original model by Alibaba
- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) - Original aligner model by Alibaba

---
