#!/bin/sh

# TreeOfLife-10M Dataset Download Script (https://huggingface.co/datasets/imageomics/TreeOfLife-10M)
# Storage: each tar ~30GB
# Downloads and extracts selected TreeOfLife image sets
# Usage: ./download.sh [set_numbers...] (1-63)
# Licensed under the Apache License, Version 2.0

BASE_URL="https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/main/dataset/EOL"

# If no arguments, default to all 63 sets
if [ $# -eq 0 ]; then
    set -- $(seq 1 63)
fi

for num in "$@"; do
    dir=$(printf "%02d" "$num")
    tarfile="image_set_${dir}.tar.gz"
    url="${BASE_URL}/${tarfile}"

    if [ -d "$dir" ]; then
        echo "Skipping $dir (already extracted)"
        continue
    fi

    echo "Downloading $tarfile ..."
    if ! wget -c "$url" -O "$tarfile"; then
        echo "Failed to download $url"
        continue
    fi

    echo "Creating directory $dir"
    mkdir -p "$dir"

    echo "Extracting $tarfile ..."
    if tar -xzf "$tarfile" -C "$dir"; then
        rm "$tarfile"
        echo "Deleted $tarfile"
    else
        echo "Extraction failed for $tarfile"
    fi
done
