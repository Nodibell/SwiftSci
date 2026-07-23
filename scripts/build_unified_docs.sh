#!/bin/bash
set -e

echo "🚀 Building Unified DocC Site for all 14 SwiftSci targets..."

# Prepare temp directory
TMP_DIR="./.build/docc_tmp"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

TARGETS=(
  "SwiftDataFrame"
  "SwiftStats"
  "SwiftPreprocessing"
  "SwiftML"
  "SwiftCluster"
  "SwiftOptimize"
  "SwiftForecast"
  "SwiftNLP"
  "SwiftExplain"
  "SwiftLLM"
  "SwiftVisualization"
  "SwiftVision"
  "SwiftDatabase"
  "SwiftAgent"
)

# 1. Build DocC archive for each target into separate temp folders
for target in "${TARGETS[@]}"; do
  echo "📦 Generating DocC for $target..."
  swift package --allow-writing-to-directory "$TMP_DIR/$target" generate-documentation \
    --target "$target" \
    --output-path "$TMP_DIR/$target" \
    --transform-for-static-hosting \
    --hosting-base-path SwiftSci
done

# 2. Use the web frontend templates from the first target (SwiftDataFrame)
BASE_TARGET="SwiftDataFrame"
rm -rf docs
mkdir -p docs/data/documentation docs/documentation docs/images docs/downloads docs/videos

cp -R "$TMP_DIR/$BASE_TARGET/css" docs/
cp -R "$TMP_DIR/$BASE_TARGET/js" docs/
cp -R "$TMP_DIR/$BASE_TARGET/img" docs/ 2>/dev/null || true
cp "$TMP_DIR/$BASE_TARGET/index.html" docs/documentation/index.html 2>/dev/null || true
cp "$TMP_DIR/$BASE_TARGET/favicon.ico" docs/ 2>/dev/null || true
cp "$TMP_DIR/$BASE_TARGET/favicon.svg" docs/ 2>/dev/null || true
cp "$TMP_DIR/$BASE_TARGET/metadata.json" docs/ 2>/dev/null || true
cp "$TMP_DIR/$BASE_TARGET/theme-settings.json" docs/ 2>/dev/null || true
touch docs/.nojekyll

# 3. Merge data/documentation and documentation for ALL 14 targets
for target in "${TARGETS[@]}"; do
  echo "🔗 Merging documentation data for $target..."
  if [ -d "$TMP_DIR/$target/data/documentation" ]; then
    cp -R "$TMP_DIR/$target/data/documentation/"* docs/data/documentation/
  fi
  if [ -d "$TMP_DIR/$target/documentation" ]; then
    cp -R "$TMP_DIR/$target/documentation/"* docs/documentation/
  fi
  if [ -d "$TMP_DIR/$target/images" ]; then
    cp -R "$TMP_DIR/$target/images/"* docs/images/ 2>/dev/null || true
  fi
done

# Clean up temp dir
rm -rf "$TMP_DIR"

echo "✅ Unified DocC site built successfully in ./docs!"
