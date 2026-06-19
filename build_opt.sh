#!/bin/bash
set -e
cd /home/nanobot/.nanobot/workspace/qwen3-asr.cpp
rm -rf build
mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/nanobot/.local/lib/python3.12/site-packages/ziglang/zig \
  -DCMAKE_C_COMPILER=/home/nanobot/.local/lib/python3.12/site-packages/ziglang/zig \
  -DCMAKE_C_COMPILER_ARG1=cc \
  -DCMAKE_CXX_COMPILER_ARG1=c++ \
  -DCMAKE_AR=/tmp/zig-ar \
  -DCMAKE_RANLIB=/tmp/zig-ranlib \
  -DCMAKE_MAKE_PROGRAM=/home/nanobot/.local/bin/ninja \
  -GNinja \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DGGML_LLAMAFILE=ON \
  -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DQWEN3_ASR_BUILD_PYTHON=OFF \
  2>&1
echo "---CMAKE_DONE---"
ninja -j$(nproc) 2>&1
echo "---BUILD_DONE---"
