# Camera-Camera Calibration on-the-fly

**Synchronize multi-camera systems on autonomous vehicles using trajectory analysis.**

This project implements a software-based synchronization solution that detects and corrects time offsets between cameras without relying on hardware triggers. It uses computer vision (ORB SLAM) and vehicle trajectory data to minimize geometric reprojection errors.

---

## 1. Project Overview

*   **Goal**: Synchronize a "Target Camera" (e.g., `CAM_FRONT_LEFT`) to a trusted "Reference Camera" (e.g., `CAM_FRONT`).
*   **Method**: Build a sparse 3D map from the reference camera, project landmarks into the target camera, and optimize the timestamp until alignment is perfect.
*   **Key Capabilities**:
    *   **Sub-millisecond accuracy**: Precise time offset recovery.
    *   **Non-overlapping FOV**: Uses vehicle motion to bridge gaps between cameras.
    *   **Validated**: Tested on the nuScenes dataset.

## 2. Architecture

The system follows a linear **Map-Then-Sync** pipeline:

1.  **Ingest**: Load raw vehicle trajectory and image data.
2.  **Map**: Use `CAM_FRONT` to build a 3D structure of the world (landmarks) using ORB features.
3.  **Match**: Project 3D landmarks into the `CAM_FRONT_LEFT` view.
4.  **Optimize**: Slide the timestamp of the target camera until the 3D points align with the image.

## 3. File Structure

### Core Modules (`src/`)

*   **`data_loader.py`**: Interface for the nuScenes dataset. Provides continuous trajectory interpolation (`pose = f(t)`) and static extrinsics.
*   **`map_maker.py`**: The "Monocular SLAM" front-end. Detects ORB features in the reference camera and triangulates them into 3D points.
*   **`matcher.py`**: Projects 3D world points into the target camera. Allows simulating time delays via a `time_offset` parameter.
*   **`optimizer.py`**: The mathematical solver. Minimizes pixel reprojection error to find the optimal time offset.

### Scripts

*   **`run_sync.py`**: Main executable. Loads data, builds the map, injects a simulated delay, and runs the optimizer to recover it.
*   **`src/test/validate_sync.py`**: Validation script. Sweeps the time offset from -100ms to +100ms to generate a "Cost Landscape" (V-Curve), proving the solution's robustness.

## 4. Getting Started

### Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data**:
    Ensure you have the nuScenes dataset installed. This project expects the data to be organized as follows:
    *   `data/v1.0-trainval`
    *   `data/samples`

### Running the Synchronization Tool

```bash
python run_sync.py
```
*Output: Reports the injected delay and the recovered offset.*

### Running Validation

```bash
python src/test/validate_sync.py
```
*Output: Displays the optimization cost landscape graph.*