#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

source ./segdac/install_dependencies.sh

pip install --no-deps -e ./segdac_dev/

pip install --no-deps -e ./segdac/

pip uninstall -y opencv-python # We use headless instead
pip uninstall -y opencv-python-headless # Remove partial install
pip install opencv-python-headless==4.11.0.86 # Re-install headless so it re-installs cv2