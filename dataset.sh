#!/bin/bash

# Ensure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi

# Create data directory if it doesn't exist
mkdir -p ../data

# A helper function to download a file.
download_file() {
    local filepath="$1"
    local cmd="$2"

    if [ -f "$filepath" ]; then
        read -p "$filepath already exists. Override? (y/n): " answer
        if [ "$answer" != "y" ]; then
            echo "Skipping download for $filepath."
            return
        fi
    fi
    echo "Downloading $filepath..."
    eval "$cmd"
}

# Download shape_data.zip using gdown
download_file "../shape_data.zip" 'gdown --fuzzy "https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link" -O ../shape_data.zip'

# Download cp2p.zip using wget
download_file "../data/cp2p.zip" 'wget -nc -O ../data/cp2p.zip "https://huggingface.co/datasets/xieyizheng/cp2p/resolve/main/CP2P.zip?download=true"'

# Download pfarm.zip using wget
download_file "../data/pfarm.zip" 'wget -nc -O ../data/pfarm.zip https://github.com/pvnieo/cp2p-pfarm-benchmark/raw/master/pfarm/pfarm.zip'

# Extract all datasets
echo "Extracting datasets..."
( cd ../ && unzip -q shape_data.zip )
( cd ../data && { unzip -q cp2p.zip; unzip -q pfarm.zip; } )

echo "All datasets downloaded and extracted successfully!"
