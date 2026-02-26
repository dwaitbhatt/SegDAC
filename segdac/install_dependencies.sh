#!/bin/bash

pip install -r ./segdac_dev/requirements.txt

pip install -r ./segdac/requirements.txt

# Use the WEIGHTS_FOLDER environment variable if set, otherwise default to "weights"
: "${WEIGHTS_FOLDER:=weights}"

if [ ! -d "$WEIGHTS_FOLDER" ]; then
    mkdir -p "$WEIGHTS_FOLDER"
    echo "Created folder: $WEIGHTS_FOLDER"
fi

# Install EfficientViT-SAM
pip install git+https://github.com/mit-han-lab/efficientvit.git@20317cb7240c81e9ded74501a523846597021133
# Download EfficientViT-SAM weights
SAM_WEIGHTS_FILE="efficientvit_sam_l0.pt"
SAM_WEIGHTS_FILE_URL="https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/$SAM_WEIGHTS_FILE"
if [ ! -f "$WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE" ]; then
    echo "Downloading $SAM_WEIGHTS_FILE..."
    wget -q -O "$WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE" "$SAM_WEIGHTS_FILE_URL"
    echo "Download complete: $WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE"
else
    echo "EfficientViT-SAM weights already exists: $WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE"
fi

# Download YOLO World weights
YOLO_WEIGHTS_FILE="yolov8s-worldv2.pt"
YOLO_WEIGHTS_FILE_URL="https://github.com/ultralytics/assets/releases/download/v8.2.0/$YOLO_WEIGHTS_FILE"
if [ ! -f "$WEIGHTS_FOLDER/$YOLO_WEIGHTS_FILE" ]; then
    echo "Downloading $YOLO_WEIGHTS_FILE..."
    wget -q -O "$WEIGHTS_FOLDER/$YOLO_WEIGHTS_FILE" "$YOLO_WEIGHTS_FILE_URL"
    echo "Download complete: $WEIGHTS_FOLDER/$YOLO_WEIGHTS_FILE"
else
    echo "YOLO World weights already exists: $WEIGHTS_FOLDER/$YOLO_WEIGHTS_FILE"
fi

# CLIP (required by YOLO)
pip install git+https://github.com/openai/CLIP.git

echo "Downloading CLIP weights..."
mkdir -p $WEIGHTS_FOLDER/clip/
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -O $WEIGHTS_FOLDER/clip/ViT-B-32.pt
echo "Download complete!"

pip uninstall -y opencv-python                # We use headless instead
pip uninstall -y opencv-python-headless       # Remove partial install (pip uninstall -y opencv-python will remove cv2 as well so we need to re-install headless)
pip install opencv-python-headless==4.11.0.86 # Re-install headless correctly
