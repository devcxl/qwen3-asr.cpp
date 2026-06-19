"""Qwen3 ASR Python bindings.

A thin pybind11 wrapper around the C++ Qwen3ASR library.
Provides load_model / transcribe (file + raw samples) with zero-copy numpy
support and RAII resource management.
"""

from ._core import Qwen3ASR, TranscribeParams, TranscribeResult, load_audio_file

__all__ = [
    "Qwen3ASR",
    "TranscribeParams",
    "TranscribeResult",
    "load_audio_file",
]
