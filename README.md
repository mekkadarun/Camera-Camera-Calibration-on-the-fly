# Visual Spatio-Temporal Synchronizer

A tool for synchronizing camera streams in autonomous driving datasets (nuScenes) using Structure-from-Motion and Deep Learning-based dynamic object masking.

## Features

- **Spatio-Temporal Synchronization**: Recovers time delays between cameras.
- **Dynamic Object Filtering**: Uses DeepLabV3 to mask out moving objects (cars, pedestrians).
- **Visual Validation**: Generates side-by-side comparison snapshots to verify synchronization.

## Installation

1. **Clone and Install**:
   ```bash
   git clone <repository_url>
   cd CameraCalibAFR
   pip install -r requirements.txt
   ```

2. **Data Setup**:
   - Place nuScenes dataset (v1.0-trainval or v1.0-mini) in `data/`.

## Usage

### 1. Run Synchronization
Runs the synchronization experiment with an injected delay and generates comparison snapshots.

```bash
python3 run_sync.py
```

**Output**:
- Console: Injected error, recovered offset, and residual error.
- `results/`: Three comparison snapshots (Frame 15).
    1. `1_desync_nomask.jpg`: Desynchronized, features on dynamic objects.
    2. `2_desync_masked.jpg`: Desynchronized, dynamic objects masked.
    3. `3_synchronized.jpg`: Synchronized (recovered), dynamic objects masked.

### 2. Generate Video Demo
Visualizes the dynamic object filtering process across a sequence.

```bash
python3 generate_masked_video.py
```
*Output: `masked_features_demo.mp4`*

## Project Structure

```
.
├── data/                   # Dataset directory
├── results/                # Output comparison snapshots
├── src/
│   ├── data_loader.py      # Data loading and interpolation
│   ├── map_maker.py        # 3D map building
│   ├── masker.py           # Dynamic object masking (DeepLabV3)
│   ├── matcher.py          # Point projection
│   ├── optimizer.py        # Time offset optimization
│   ├── visualizer.py       # Visualization tools
│   └── test/               # Unit tests
├── run_sync.py             # Main synchronization script
├── generate_masked_video.py# Video generation script
└── requirements.txt        # Dependencies
```