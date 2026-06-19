#!/usr/bin/env bash
set -euo pipefail

CC="python3 -m ziglang cc"
CXX="python3 -m ziglang c++"
AR="python3 -m ziglang ar"

BASE_DIR="$PWD"
BUILD_DIR="$PWD/build"
GGML_DIR="$PWD/ggml"
GGML_BUILD="$GGML_DIR/build"

mkdir -p "$BUILD_DIR"
mkdir -p "$GGML_BUILD"

GGML_CFLAGS="-O0 -fPIC -DGGML_CUDA=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -include $BASE_DIR/ggml-config.h"
GGML_CFLAGS="$GGML_CFLAGS -I$GGML_DIR/include -I$GGML_DIR -I$BASE_DIR"

CXXFLAGS="-std=c++17 -O0 -fPIC -DGGML_CUDA=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF"
CXXFLAGS="$CXXFLAGS -I$GGML_DIR/include -I$GGML_DIR -I$BASE_DIR -I$BASE_DIR/src -I$BASE_DIR/third_party"

echo "=== Step 1: Build ggml static libs ==="

GGML_SRCS="ggml ggml-alloc ggml-base ggml-quants"
GGML_OBJS=""
for name in $GGML_SRCS; do
    src="$GGML_DIR/src/$name.c"
    obj="$GGML_BUILD/$name.o"
    echo "  CC $name.c"
    $CC -c $GGML_CFLAGS "$src" -o "$obj"
    GGML_OBJS="$GGML_OBJS $obj"
done

# ggml-cpu needs -x c (force C mode with zig's clang)
src="$GGML_DIR/src/ggml-cpu.c"
obj="$GGML_BUILD/ggml-cpu.o"
echo "  CC ggml-cpu.c"
$CC -c $GGML_CFLAGS "$src" -o "$obj"
GGML_OBJS="$GGML_OBJS $obj"

# ggml-backend needs -x c
src="$GGML_DIR/src/ggml-backend.c"
obj="$GGML_BUILD/ggml-backend.o"
echo "  CC ggml-backend.c"
$CC -c $GGML_CFLAGS "$src" -o "$obj"
GGML_OBJS="$GGML_OBJS $obj"

echo "  AR libggml.a"
$AR rcs "$GGML_BUILD/libggml.a" $GGML_OBJS

echo "=== Step 2: Build project source files ==="

SRC_NAMES="mel_spectrogram gguf_loader audio_encoder text_decoder audio_injection qwen3_asr"
SRC_OBJS=""
for name in $SRC_NAMES; do
    src="$BASE_DIR/src/$name.cpp"
    obj="$BUILD_DIR/$name.o"
    echo "  CXX $name.cpp"
    $CXX -c $CXXFLAGS "$src" -o "$obj"
    SRC_OBJS="$SRC_OBJS $obj"
done

echo "=== Step 3: Build qwen3-asr-server ==="
$CXX $CXXFLAGS "$BASE_DIR/src/server.cpp" $SRC_OBJS "$GGML_BUILD/libggml.a" -lpthread -lm -o "$BUILD_DIR/qwen3-asr-server"

echo ""
echo "=== Build complete! ==="
echo "Server: $BUILD_DIR/qwen3-asr-server"
