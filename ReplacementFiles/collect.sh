#!/bin/bash
# Usage: ./collect_my_witness.sh <ns3-source-root> <destination-dir>

SRC_DIR="$1"
DST_DIR="$2"

if [ -z "$SRC_DIR" ] || [ -z "$DST_DIR" ]; then
  echo "Usage: $0 <source_dir> <destination_dir>"
  exit 1
fi

# Make sure the destination exists
mkdir -p "$DST_DIR"

# Find all files (any type) under ns-3 source that contain "mahdi"
# Adjust pattern if you used a different signature (case-sensitive)
grep -rli --exclude-dir='build' --exclude-dir='out' "mahdi" "$SRC_DIR" | while read -r file; do
    # Extract just the filename
  filename="$(basename "$file")"

  # Copy the file (overwrite if exists, preserve timestamps and permissions)
  cp -pf "$file" "$DST_DIR/$filename"

  echo "Copied: $file → $DST_DIR/$filename"
done

echo "Done. Files containing 'mahdi' have been copied to: $DST_DIR"
