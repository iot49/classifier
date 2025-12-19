#!/bin/bash
set -e

# Default model name
MODEL_NAME="resnet18"
TAG_VERSION=""

# Parsing arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-name) MODEL_NAME="$2"; shift ;;
        --tag) TAG_VERSION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check dependencies
if ! command -v gh &> /dev/null; then
    echo "Error: gh (GitHub CLI) is not installed."
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "Error: python is not installed."
    exit 1
fi

echo "--- Releasing Model: $MODEL_NAME ---"

# Step 1: Export Model
echo "Running export..."
python main.py --export --model-name "$MODEL_NAME"

ONNX_FILE="models/onnx/${MODEL_NAME}.onnx"
CONFIG_FILE="models/onnx/${MODEL_NAME}.config.json"

if [[ ! -f "$ONNX_FILE" ]]; then
    echo "Error: Expected ONNX file not found at $ONNX_FILE"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Expected config file not found at $CONFIG_FILE"
    exit 1
fi

echo "Files ready:"
ls -lh "$ONNX_FILE" "$CONFIG_FILE"

# Step 2: Create Release
if [[ -z "$TAG_VERSION" ]]; then
    read -p "Enter release tag (e.g., v0.1.0): " TAG_VERSION
fi

if [[ -z "$TAG_VERSION" ]]; then
    echo "No tag provided. specificy --tag or enter when prompted."
    exit 1
fi

read -p "Create release $TAG_VERSION and upload assets? (y/N) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo "Creating GitHub Release $TAG_VERSION..."
gh release create "$TAG_VERSION" "$ONNX_FILE" "$CONFIG_FILE" --title "Model Release $TAG_VERSION" --notes "Automated release of $MODEL_NAME"

echo "Success! Release $TAG_VERSION created."
