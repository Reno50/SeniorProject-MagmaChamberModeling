#!/bin/bash

# Default output subfolder if none is provided
DEFAULT_SUBFOLDER="NewEquationsModel"
OUTPUT_ROOT="outputs"

# Use argument if provided, else fallback
SUBFOLDER="${1:-$DEFAULT_SUBFOLDER}"
TARGET_DIR="$OUTPUT_ROOT/$SUBFOLDER"

# Ensure the directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' does not exist."
  exit 1
fi

echo "Working in: $TARGET_DIR"

# Remove .hydra and constraints directories if they exist
rm -rf "$TARGET_DIR/.hydra" "$TARGET_DIR/constraints"

# Remove all files and directories in $TARGET_DIR except 'validators'
find "$TARGET_DIR" -mindepth 1 -maxdepth 1 ! -name 'validators' -exec rm -rf {} +

# Ensure validators directory exists
VALIDATORS_DIR="$TARGET_DIR/validators"
if [ ! -d "$VALIDATORS_DIR" ]; then
  echo "No validators directory found. Exiting."
  exit 0
fi

# Generate a unique prefix using current timestamp
PREFIX="run_$(date +%Y%m%d_%H%M%S)"

# Rename all image files in validators directory
shopt -s nullglob
for img in "$VALIDATORS_DIR"/*.png "$VALIDATORS_DIR"/*.jpg "$VALIDATORS_DIR"/*.jpeg; do
  base=$(basename "$img")
  mv "$img" "$VALIDATORS_DIR/${PREFIX}_$base"
done
shopt -u nullglob

echo "Cleanup and renaming complete. Images now prefixed with: $PREFIX"
