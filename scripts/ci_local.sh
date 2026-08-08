#!/usr/bin/env bash
set -e

echo "🚀 Running SwiftSci CI locally on macOS (with full Apple Silicon Metal GPU access)..."

echo "📦 1. Building SwiftSci (Debug mode)..."
swift build -v

echo "📦 2. Building SwiftSci (Release mode)..."
swift build -c release

echo "🧪 3. Running ALL Unit Tests (including MLX GPU & Metal accelerated suites)..."
swift test --enable-code-coverage

echo "📚 4. Checking DocC Documentation Warnings..."
swift package generate-documentation \
  --target SwiftDataFrame \
  --target SwiftStats \
  --target SwiftPreprocessing \
  --target SwiftML \
  --target SwiftCluster \
  --target SwiftNLP \
  --target SwiftOptimize \
  --target SwiftForecast \
  --target SwiftLLM \
  --target SwiftExplain \
  --target SwiftVisualization \
  --target SwiftVision \
  --target SwiftDatabase \
  --target SwiftAgent \
  --warnings-as-errors

echo "📊 5. Verifying Public API Documentation Coverage..."
python3 scripts/verify_doc_coverage.py

echo "✅ All CI checks + GPU Metal suites passed cleanly on your local Mac!"
