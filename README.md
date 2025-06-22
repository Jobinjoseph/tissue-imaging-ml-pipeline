# Tissue Imaging ML Pipeline

This project implements a simple deep learning pipeline for tissue image analysis using Python and TensorFlow. It preprocesses large tissue images by splitting them into patches, trains a CNN classifier on those patches, and visualizes prediction heatmaps overlayed on the original tissue.

## Features

- Image loading and preprocessing (tiling WSIs into 128x128 patches)
- Simple CNN architecture for patch classification
- Visualization of predicted class heatmaps
- AWS S3 integration for uploading/downloading images and models
- Modular and extensible codebase for translational pathology research

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/tissue-imaging-ml-pipeline.git
   cd tissue-imaging-ml-pipeline
