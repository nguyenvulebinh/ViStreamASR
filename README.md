# 🎙️ ViStreamASR - Real-Time Vietnamese Speech Recognition

**ViStreamASR** is a simple Vietnamese Streaming Automatic Speech Recognition library for real-time audio processing.

## Features

- 🎯 **Streaming ASR**: Real-time audio processing with configurable chunk sizes
- 🇻🇳 **Vietnamese Optimized**: Specifically designed for Vietnamese speech recognition
- 📦 **Simple API**: Easy-to-use interface with minimal setup
- ⚡ **High Performance**: CPU/GPU support

## Installation

```bash
pip install ViStreamASR
```

## Installation from Source

For development or to use the latest version:

```bash
# Clone the repository
git clone https://github.com/nguyenvulebinh/ViStreamASR.git
cd ViStreamASR

# Install dependencies
pip install -r requirements.txt

# Option 1: Use directly from source
python test_library.py  # Test the installation

# Option 2: Install in development mode
pip install -e .
```

### Using from Source

When using from source, import the modules directly:

```python
import sys
sys.path.insert(0, 'src')
from streaming import StreamingASR

# Initialize and use
asr = StreamingASR()
for result in asr.stream_from_file("audio.wav"):
    print(result['text'])
```

## Quick Start

### Python API

#### Streaming from File
```python
from ViStreamASR import StreamingASR

# Initialize ASR
asr = StreamingASR()

# Process audio file
for result in asr.stream_from_file("audio.wav"):
    if result['partial']:
        print(f"Partial: {result['text']}")
    if result['final']:
        print(f"Final: {result['text']}")
```

#### Streaming from Microphone
```python
from ViStreamASR import StreamingASR

# Initialize ASR
asr = StreamingASR()

# Process audio file
for result in asr.stream_from_microphone(duration_seconds=10):
    if result['partial']:
        print(f"Partial: {result['text']}")
    if result['final']:
        print(f"Final: {result['text']}")
```

### Command Line

```bash
# Basic transcription
vistream-asr transcribe audio.wav
```

## API Reference

### StreamingASR

```python
from ViStreamASR import StreamingASR

# Initialize with options
asr = StreamingASR(
    chunk_size_ms=640,           # Chunk size in milliseconds
    auto_finalize_after=15.0,    # Auto-finalize after seconds
    debug=False                  # Enable debug logging
)

# Stream from file
for result in asr.stream_from_file("audio.wav"):
    # result contains:
    # - 'partial': True for partial results
    # - 'final': True for final results
    # - 'text': transcription text
    # - 'chunk_info': processing information
    pass
```

### Advanced Usage

For low-level control:

```python
from ViStreamASR import ASREngine

engine = ASREngine(chunk_size_ms=640, debug_mode=True)
engine.initialize_models()

# Process audio chunks directly
result = engine.process_audio(audio_chunk, is_last=False)
```

## Model Information

- **Language**: Vietnamese
- **Architecture**: [U2-based](https://arxiv.org/abs/2203.15455) streaming ASR
- **Model Size**: ~2.7GB (cached after first download)
- **Sample Rate**: 16kHz (automatically converted)
- **Optimal Chunk Size**: 640ms

### How U2 Streaming Works

The following picture shows how U2 (Unified Streaming and Non-streaming) architecture works:

![U2 Architecture](resource/u2.gif)

The U2 model enables both streaming and non-streaming ASR in a unified framework, providing low-latency real-time transcription while maintaining high accuracy.

## Performance

- **RTF**: ~0.34x (faster than real-time)
- **Latency**: ~640ms with default settings
- **GPU Support**: Automatic CUDA acceleration when available

## Limitations

- **Audio Input Assumption**: The system assumes audio input is speech. Non-speech audio may produce unexpected results.
- **Production Recommendation**: For practical use, it's recommended to add VAD (Voice Activity Detection) before running streaming ASR to reduce ASR streaming load and improve efficiency.

## CLI Commands

```bash
# Transcription
vistream-asr transcribe <file>                    # Basic transcription
vistream-asr transcribe <file> --chunk-size 640   # Custom chunk size
vistream-asr transcribe <file> --no-debug         # Clean output

# Information
vistream-asr info                                  # Library info
vistream-asr version                               # Version
```

## Requirements

### System Requirements

- **RAM**: Minimum 5GB RAM
- **CPU**: Minimum 2 cores
- **Performance**: RTF 0.3-0.4x achievable on CPU-only systems meeting above specs
- **GPU**: Supports GPU acceleration for better performance, but CPU-only operation still achieves RTF 0.3-0.4x

### Software Requirements

- Python 3.8+
- PyTorch 2.5+
- TorchAudio 2.5+
- NumPy 1.19.0+
- Requests 2.25.0+
- flashlight-text
- librosa

## License

MIT License