#!/usr/bin/env bash
# Build the two Lambda packages for arm64 (Graviton) and Python 3.13.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build && mkdir -p build/analyzer build/api
python3 -m pip install --quiet --target build/analyzer \
  --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 --only-binary=:all: --implementation cp --python-version 3.13 \
  opencv-python-headless==5.0.0.93 numpy==2.4.6
cp -R ../sightline build/analyzer/sightline
rm -rf build/analyzer/sightline/__pycache__
cp analyzer/handler.py build/analyzer/handler.py
cp api/handler.py api/index.html build/api/
(cd build/analyzer && zip -qr9 ../analyzer.zip .)
(cd build/api && zip -qr9 ../api.zip .)
ls -l build/*.zip | awk '{print $5, $9}'
