#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Create data directory if it doesn't exist
mkdir -p ../data

# Download datasets using gdown
echo "Downloading datasets..."
gdown --fuzzy "https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link" -O ../shape_data.zip

# Download cp2p dataset
echo "Downloading cp2p dataset..."
wget -O ../data/cp2p.zip "https://huggingface.co/datasets/xieyizheng/cp2p/resolve/main/CP2P.zip?download=true"

# Download pfarm dataset
echo "Downloading pfarm dataset..."
wget -O ../data/pfarm.zip https://github.com/pvnieo/cp2p-pfarm-benchmark/raw/master/pfarm/pfarm.zip

# Extract all datasets
echo "Extracting datasets..."
cd ../
unzip -q shape_data.zip
cd data
unzip -q cp2p.zip
unzip -q pfarm.zip

# Clean up zip files
# rm ../shape_data.zip cp2p.zip pfarm.zip

echo "All datasets downloaded and extracted successfully!"
