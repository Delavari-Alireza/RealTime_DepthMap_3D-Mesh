# RealTime DepthMap and 3d Mesh reconstruction based on DepthAnything V2

This repository provides a set of Python scripts demonstrating how to utilize the DepthAnything V2 model for depth estimation and 3D reconstruction from images and videos. These examples leverage the `transformers` library and Open3D to create depth maps, point clouds, and 3D meshes.

![DepthMap VIDEO](pictures/output.gif "DepthMap VIDEO")
![3D_MESH_VIDEO](pictures/mesh_output.gif "3D_MESH_VIDEO")


## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Scripts Overview](#scripts-overview)
5. [Usage](#usage)
6. [References](#references)

---

## Introduction

DepthAnything V2 is an advanced depth estimation model that generates accurate depth maps from 2D images. This repository demonstrates its capabilities through Python scripts for various use cases, including:

- Depth map generation from images.
- Depth map visualization.
- Conversion of images to 3D point clouds and meshes.
- Depth estimation from videos with real-time visualization.

---

## Prerequisites

Ensure the following requirements are met:

1. **Operating System:** Linux or Windows.
2. **GPU Support:** NVIDIA CUDA-compatible GPU.
3. **Dependencies:** Python 3.8+ with required libraries.
4. **DepthAnyThingV2 Pre-trained Models**: Donwload the pre-trained Models from [here](https://github.com/DepthAnything/Depth-Anything-V2)

---

## Installation

1. Clone this repository:
    ```bash
    git clone --recursive https://github.com/Delavari-Alireza/DepthAnythingV2_Demo.git
    cd DepthAnythingV2_Demo
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Verify GPU setup for PyTorch:
    ```python
    import torch
    print(torch.cuda.is_available())  # Should return True
    ```

---

## Scripts Overview

### 1. **Image to 3D Mesh (HuggingFace)**
   - Script: `ImageTo3DMesh_HuggingFace.py`
   - Description: Uses the DepthAnything V2 HuggingFace model to generate depth maps and convert them into 3D meshes.

### 2. **Image to 3D Mesh (Source Code)**
   - Script: `ImageTo3DMesh_SourceCode.py`
   - Description: Demonstrates depth map generation using the DepthAnything V2 source code.

### 3. **Image to Depth Map (HuggingFace)**
   - Script: `ImageToDepthMap_HuggingFace.py`
   - Description: Generates depth maps for single images using the HuggingFace pipeline.

### 4. **Image to Depth Map (Source Code)**
   - Script: `ImageToDepthMap_SourceCode.py`
   - Description: Uses DepthAnything V2 source code to generate depth maps from images.

### 5. **Video to 3D Mesh (HuggingFace)**
   - Script: `VideoTo3DMesh_HuggingFace.py`
   - Description: Processes videos to generate 3D meshes frame by frame.

---

## Usage

### Running the Scripts

1. **Image to Depth Map**
   ```bash
   python ImageToDepthMap_HuggingFace.py --img-path /path/to/image.jpg
   ```

2. **Image to 3D Mesh**
   ```bash
   python ImageTo3DMesh_HuggingFace.py --img-path /path/to/image.jpg
   ```

3. **Video to 3D Mesh**
   ```bash
   python VideoTo3DMesh_HuggingFace.py --video-path /path/to/video.mp4
   ```

### Key Parameters

- `--img-path`: Path to the input image.
- `--video-path`: Path to the input video file.
- `--focal-length-x` and `--focal-length-y`: Camera intrinsic parameters for point cloud generation.

---

## References

- [DepthAnything V2 GitHub Repository](https://github.com/DepthAnything/Depth-Anything-V2)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Open3D Documentation](http://www.open3d.org/docs/latest/)
- [Free stock footage by Videezy](http://www.videezy.com/)
---

For further questions or contributions, feel free to open an issue or submit a pull request!

