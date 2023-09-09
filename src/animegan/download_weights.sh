#!/bin/bash

DOWNLOAD_DIR="$(pwd)/weights"

declare -a URLs=(
https://github.com/kangnam7654/animegan2-pytorch/raw/main/weights/celeba_distill.pt
https://github.com/kangnam7654/animegan2-pytorch/raw/main/weights/face_paint_512_v1.pt
"https://github.com/kangnam7654/animegan2-pytorch/raw/main/weights/face_paint_512_v2.pt"
"https://github.com/kangnam7654/animegan2-pytorch/raw/main/weights/paprika.pt"
)

mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

for URL in "${URLs[@]}"; do
    wget $URL
done