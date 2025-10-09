#!/bin/bash
# Usage: ./replace_in_source.sh <ns3-source-root> <modified-files-dir>
# Example: ./replace_in_source.sh ~/Documents/ns-allinone-3.41/ns-3.41 ~/witness_files

SRC_DIR="$1"
MOD_DIR="$2"

if [ -z "$SRC_DIR" ] || [ -z "$MOD_DIR" ]; then
  echo "Usage: $0 <source_dir> <modified_files_dir>"
  exit 1
fi

# Verify directories exist
if [ ! -d "$SRC_DIR" ]; then
  echo "Error: source directory not found: $SRC_DIR"
  exit 1
fi

if [ ! -d "$MOD_DIR" ]; then
  echo "Error: modified files directory not found: $MOD_DIR"
  exit 1
fi

# Iterate over every file in the modified directory
for mod_file in "$MOD_DIR"/*; do
  filename="$(basename "$mod_file")"

  # Find all files in SRC_DIR matching this filename
  matches=($(find "$SRC_DIR" -type f -name "$filename" ! -path "*/build/*" ! -path "*/out/*"))

  if [ ${#matches[@]} -eq 0 ]; then
    echo "⚠️  No match found for: $filename"
  elif [ ${#matches[@]} -eq 1 ]; then
    # Single match → overwrite directly
    cp -pf "$mod_file" "${matches[0]}"
    echo "✅ Replaced: ${matches[0]}"
  else
    # Multiple matches → warn but still overwrite all
    echo "⚠️  Multiple matches for $filename:"
    for m in "${matches[@]}"; do
      cp -pf "$mod_file" "$m"
      echo "   → Replaced: $m"
    done
  fi
done

echo "✅ Done. All matching files in '$SRC_DIR' have been replaced from '$MOD_DIR'."