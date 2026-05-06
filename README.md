# Visual SLAM Research Project

A Visual SLAM (Simultaneous Localization and Mapping) project focused on building a system that estimates camera motion while constructing a map of the environment from visual input. The goal is to support localization in GNSS-denied environments using computer vision, feature tracking, and geometric estimation.

## Description

This project explores the design and implementation of a Visual SLAM pipeline for research and experimentation. It aims to recover camera pose over time and build a sparse map of the surrounding scene from image sequences. The system is useful for robotics, autonomous navigation, and visual localization in environments where GPS is unavailable or unreliable.

The project include components such as:

* image preprocessing
* feature detection and matching
* camera motion estimation
* triangulation and map point generation
* pose tracking
* loop closure and map refinement
* visualization of trajectory and reconstructed landmarks

## Features

* Monocular visual odometry / SLAM pipeline
* Feature extraction and tracking
* Camera pose estimation
* Incremental map building
* Trajectory visualization
* Modular codebase for experimentation and research

## Requirements

### System Requirements

* Linux, macOS, or Windows
* Python 3.10+
* Git
* A camera, dataset, or image sequence for testing

### Python Dependencies

If the project is implemented in Python, install the following packages:

```bash
numpy
opencv-python
matplotlib
scipy
```

Additional packages may be required

```bash
jupyter
plotly
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Mvoii/vo_kitti_3.git
cd vo_kitti_3
```

### Python Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main application with a dataset or image sequence path:

### Python

```bash
# run purely visual odometry model
python3 VO.py

# run stero visual odemetry with bundle adjeustment
python3 stereo_visual_odometry_w_bundle_adjustment.py
```

The system typically expects one of the following:

* a folder containing sequential camera images, the repo has 50 frames of the kitti sequence 1
* a monocular dataset such as an academic benchmark sequence

## Output

The project produce:

* estimated camera trajectory
* tracked feature points
* sparse 3D map points
* debug logs
* visualization graphs of the ground truth pose against the estimated poses

## Project Structure

```text
vo_kitti_3/
|--mod
|    |--VisualSLAM
|        |-- KITTI_sequence_1
|        |-- KITTI_sequence_2
|        |-- lib
|        |    |-- __init__.py
|        |    |-- visualization
|        |          |-- __init__.py
|        |          |-- camera.py
|        |          |-- image.py
|        |          |-- plotting.ppy
|        |          |-- video.py
|        |
|        |-- b_adj.txt
|        |-- bag_of_words.py
|        |-- bundle_adjustment.py
|        |-- data_structure.py
|        |-- stereo_visual_odometry_w_bundle_adjustment.py
|        |-- stereoSLAMLoopDetection.py
|        |-- VO.py
|
|--org
    |--VisualSLAM
        |-- KITTI_sequence_1
        |-- KITTI_sequence_2
        |-- lib
        |    |-- __init__.py
        |    |-- visualization
        |          |-- __init__.py
        |          |-- camera.py
        |          |-- image.py
        |          |-- plotting.ppy
        |          |-- video.py
        |
        |-- b_adj.txt
        |-- bag_of_words.py
        |-- bundle_adjustment.py
        |-- data_structure.py
        |-- stereo_visual_odometry_w_bundle_adjustment.py
        |-- stereoSLAMLoopDetection.py
        |-- VO.py
```

## Notes

* This project is intended for research and learning purposes.
* Results depend heavily on camera calibration, image quality, lighting, and motion blur.
* For best performance, use well-synchronized and properly calibrated image data.

## Future Improvements

* Loop closure detection
* Bundle adjustment
* Better feature matching and outlier rejection
* Support for stereo or RGB-D inputs
* Improved visualization and evaluation tools
* Integration with robotic navigation frameworks

## Author
Franklin Mvoi - mvoifranklin@gmail.com

Developed as part of a research dissertation on visual localization and mapping in GNSS-denied environments.
