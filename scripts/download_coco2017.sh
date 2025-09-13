#!/bin/bash

# COCO 2017 Dataset Download Script (https://cocodataset.org/#download)
# Storage: ~20GB (train+val) - Note: Temporarily requires ~2x space during zip extraction
# Downloads COCO 2017 dataset (detection only) split with images and annotations
# Usage: ./download_coco2017.sh [train|val|test|unlabeled|all] [--no-extract]
# Licensed under the Apache License, Version 2.0

set -e  # Exit on any error

DATASET_DIR="cocodataset"
BASE_URL_IMG="http://images.cocodataset.org/zips"
BASE_URL_ANN="http://images.cocodataset.org/annotations"

declare -A FILES=(
    ["train2017.zip"]="cced6f7f71b7629ddf16f17bbcfab6b2"
    ["val2017.zip"]="442b8da7639aecaf257c1dceb8ba8c80"
    ["test2017.zip"]="77ad2c53ac5d0aea611d422c0938fb35"
    ["unlabeled2017.zip"]="7ebc562819fdb32847aab79530457326"
    ["annotations_trainval2017.zip"]="f4bbac642086de4f52a3fdda2de5fa2c"
    ["image_info_test2017.zip"]="85da7065e5e600ebfee8af1edb634eb5"
    ["image_info_unlabeled2017.zip"]="ede38355d5c3e5251bb7f8b68e2c068f"
)

usage() {
    echo "Usage: $0 [train|val|test|unlabeled|all]"
    echo "  train     - Download training images and annotations"
    echo "  val       - Download validation images and annotations"
    echo "  test      - Download test images and annotations"
    echo "  unlabeled - Download unlabeled images and annotations"
    echo "  all       - Download all splits (default)"
    echo ""
    echo "Options:"
    echo "  --no-extract  - Skip extraction of downloaded files"
    exit 1
}

verify_checksum() {
    local file="$1"
    local file_path="$2"
    local expected_md5="${FILES[$file]}"

    if [[ -z "$expected_md5" ]]; then
        echo "Warning: No checksum for $file"
        return 0
    fi

    echo "Verifying checksum for $file..."
    local actual_md5
    if command -v md5sum &> /dev/null; then
        actual_md5=$(md5sum "$file_path" | cut -d' ' -f1)
    elif command -v md5 &> /dev/null; then
        actual_md5=$(md5 -q "$file_path")
    else
        echo "Warning: No MD5 tool found, skipping verification"
        return 0
    fi

    if [[ "$actual_md5" == "$expected_md5" ]]; then
        echo "Checksum verified for $file"
        return 0
    else
        echo "Checksum failed for $file"
        echo "  Expected: $expected_md5"
        echo "  Got:      $actual_md5"
        return 1
    fi
}

extract_file() {
    local file_path="$1"
    local extract_dir="$2"

    echo "Extracting $(basename "$file_path")..."

    mkdir -p "$extract_dir"
    if command -v unzip &> /dev/null; then
        unzip -q "$file_path" -d "$extract_dir"
        echo "Successfully extracted $(basename "$file_path")"
    else
        echo "Error: unzip is required but not installed"
        return 1
    fi
}

download_file() {
    local file="$1"
    local url="$2"
    local dest="$3"

    echo "Downloading $file..."
    wget -c "$url" -O "$dest"

    if ! verify_checksum "$file" "$dest"; then
        echo "Error: Failed to verify $file"
        return 1
    fi

    # Extract the file if extraction is enabled
    if [[ "$EXTRACT" == true ]]; then
        extract_file "$dest" "$DATASET_DIR"
        # rm "$dest"
    fi
}

download_split() {
    local split="$1"

    echo "Downloading COCO 2017 $split split..."
    mkdir -p "$DATASET_DIR"

    case $split in
        train)
            download_file "train2017.zip" "$BASE_URL_IMG/train2017.zip" "$DATASET_DIR/train2017.zip"
            download_file "annotations_trainval2017.zip" "$BASE_URL_ANN/annotations_trainval2017.zip" "$DATASET_DIR/annotations_trainval2017.zip"
            ;;
        val)
            download_file "val2017.zip" "$BASE_URL_IMG/val2017.zip" "$DATASET_DIR/val2017.zip"
            download_file "annotations_trainval2017.zip" "$BASE_URL_ANN/annotations_trainval2017.zip" "$DATASET_DIR/annotations_trainval2017.zip"
            ;;
        test)
            download_file "test2017.zip" "$BASE_URL_IMG/test2017.zip" "$DATASET_DIR/test2017.zip"
            download_file "image_info_test2017.zip" "$BASE_URL_ANN/image_info_test2017.zip" "$DATASET_DIR/image_info_test2017.zip"
            ;;
        unlabeled)
            download_file "unlabeled2017.zip" "$BASE_URL_IMG/unlabeled2017.zip" "$DATASET_DIR/unlabeled2017.zip"
            download_file "image_info_unlabeled2017.zip" "$BASE_URL_ANN/image_info_unlabeled2017.zip" "$DATASET_DIR/image_info_unlabeled2017.zip"
            ;;
        all)
            download_split "train"
            download_split "val" 
            download_split "test"
            download_split "unlabeled"
            return
            ;;
        *)
            echo "Error: Invalid split '$split'"
            usage
            ;;
    esac

    echo "$split split download completed"
}

main() {
    SPLIT="all"
    EXTRACT=true
    while [[ $# -gt 0 ]]; do
        case $1 in
            train|val|test|unlabeled|all)
                SPLIT="$1"
                shift
                ;;
            --no-extract)
                EXTRACT=false
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "Error: Unknown argument '$1'"
                usage
                ;;
        esac
    done

    if ! command -v wget &> /dev/null; then
        echo "Error: wget is required but not installed"
        exit 1
    fi

    if [[ "$EXTRACT" == true ]] && ! command -v unzip &> /dev/null; then
        echo "Error: unzip is required for extraction but not installed."
        exit 1
    fi

    echo "Starting COCO 2017 dataset download..."
    echo "Split: $SPLIT"
    echo "Dataset directory: $DATASET_DIR"
    echo "Extract files: $EXTRACT"
    echo ""

    # Download split
    download_split "$SPLIT"

    echo ""
    echo "Download completed successfully"
    echo "Files saved to: $(pwd)/$DATASET_DIR"
}

# Run main function
main "$@"
