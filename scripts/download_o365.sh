#!/bin/bash

# Objects365-2020 Dataset Download Script (https://www.objects365.org/overview.html)
# Storage: ~360GB (train+val) - Note: Temporarily requires ~2x space during tar extraction
# Downloads the Objects365 dataset splits (train, val, test, or all)
# Usage: ./download_o365.sh [train|val|test|all]
# Licensed under the Apache License, Version 2.0

set -e  # Exit on any error

DATASET_DIR="Objects365-2020"
BASE_URL="https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"

usage() {
    echo "Usage: $0 [train|val|test|all]"
    echo "  train - Download only training data"
    echo "  val   - Download only validation data"
    echo "  test  - Download only test data"
    echo "  all   - Download all splits (default)"
    exit 1
}

create_directories() {
    echo "Creating directory structure..."
    mkdir -p "${DATASET_DIR}/license/"
    mkdir -p "${DATASET_DIR}/train/"
    mkdir -p "${DATASET_DIR}/val/images/v1/"
    mkdir -p "${DATASET_DIR}/val/images/v2/"
    mkdir -p "${DATASET_DIR}/test/images/v1/"
    mkdir -p "${DATASET_DIR}/test/images/v2/"
    echo "Directories created successfully."
}

download_license() {
    echo "Downloading license..."
    wget -c "${BASE_URL}/license/license.txt.tar.gz" -P "${DATASET_DIR}/license/"
    echo "License downloaded."
}

download_train() {
    echo "Downloading training data..."

    wget -c "${BASE_URL}/train/zhiyuan_objv2_train.tar.gz" -P "${DATASET_DIR}/train/"

    echo "Downloading training image patches (0-50)..."
    for i in {0..50}; do
        echo "Downloading training patch ${i}/50..."
        wget -c "${BASE_URL}/train/patch${i}.tar.gz" -P "${DATASET_DIR}/train/"
    done

    echo "Training data download completed."
    echo "Removing known corrupted file"
    rm "${DATASET_DIR}/train/images/v1/patch4/objects365_v1_00215749.jpg.ddEA04F9"
}

download_val() {
    echo "Downloading validation data..."

    wget -c "${BASE_URL}/val/zhiyuan_objv2_val.json" -P "${DATASET_DIR}/val/"
    wget -c "${BASE_URL}/val/sample_2020.json.tar.gz" -P "${DATASET_DIR}/val/"

    echo "Downloading validation v1 image patches (0-15)..."
    for i in {0..15}; do
        echo "Downloading validation v1 patch ${i}/15..."
        wget -c "${BASE_URL}/val/images/v1/patch${i}.tar.gz" -P "${DATASET_DIR}/val/images/v1/"
    done

    echo "Downloading validation v2 image patches (16-43)..."
    for i in {16..43}; do
        echo "Downloading validation v2 patch ${i}/43..."
        wget -c "${BASE_URL}/val/images/v2/patch${i}.tar.gz" -P "${DATASET_DIR}/val/images/v2/"
    done

    echo "Validation data download completed."
}

download_test() {
    echo "Downloading test data..."

    echo "Downloading test v1 image patches (0-15)..."
    for i in {0..15}; do
        echo "Downloading test v1 patch ${i}/15..."
        wget -c "${BASE_URL}/test/images/v1/patch${i}.tar.gz" -P "${DATASET_DIR}/test/images/v1/"
    done

    echo "Downloading test v2 image patches (16-50)..."
    for i in {16..50}; do
        echo "Downloading test v2 patch ${i}/50..."
        wget -c "${BASE_URL}/test/images/v2/patch${i}.tar.gz" -P "${DATASET_DIR}/test/images/v2/"
    done

    echo "Test data download completed."
}

main() {
    # Check if wget is available
    if ! command -v wget &> /dev/null; then
        echo "Error: wget is required but not installed."
        exit 1
    fi

    # Parse command line argument
    SPLIT=${1:-"all"}

    # Validate argument
    case $SPLIT in
        train|val|test|all)
            ;;
        *)
            echo "Error: Invalid argument '$SPLIT'"
            usage
            ;;
    esac

    echo "Starting Objects365-2020 dataset download..."
    echo "Split: $SPLIT"
    echo "Dataset directory: $DATASET_DIR"
    echo ""

    # Create directory structure
    create_directories

    # Download license
    download_license

    # Download requested split
    case $SPLIT in
        train)
            download_train
            ;;
        val)
            download_val
            ;;
        test)
            download_test
            ;;
        all)
            download_train
            download_val
            download_test
            ;;
    esac

    echo ""
    echo "Download completed successfully!"
    echo "Dataset location: $(pwd)"
}

main "$@"
