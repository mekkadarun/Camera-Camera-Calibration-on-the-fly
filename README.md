Spatio-Temporal Camera Synchronization

1. Project Overview

This project implements a solution to synchronize multi-camera systems on autonomous vehicles. Instead of relying on hardware triggers, it uses computer vision and trajectory analysis to detect time offsets.

It works by building a sparse 3D map using a trusted "Reference Camera" and then optimizing the timestamp of a "Target Camera" until the geometric reprojection error is minimized.

Key capabilities:

Sub-millisecond synchronization accuracy.

Works with non-overlapping fields of view (using vehicle motion to bridge the gap).

Validated using the nuScenes dataset.

2. Functional Architecture

The system follows a linear "Map-Then-Sync" pipeline:

Ingest: Load raw vehicle trajectory and image data.

Map: Use the trusted CAM_FRONT to build a 3D structure of the world (landmarks).

Match: Project those 3D landmarks into the CAM_FRONT_LEFT view.

Optimize: Slide the timestamp of the side camera forward/backward until the 3D points line up perfectly with the image.

3. File Structure & Script Descriptions

data_loader.py (The Interface)

Role: Hides the complexity of the nuScenes dataset.

Key Function: get_trajectory_interpolator(scene). It returns a continuous function pose = f(t) that allows us to query the car's position at any microsecond, not just when images were taken.

Calibration: Also handles retrieving the static extrinsic transforms ($T_{body \to camera}$).

map_maker.py (The Reference)

Role: Acts as a "Monocular SLAM" front-end.

Key Logic: It takes pairs of images from the Reference Camera (CAM_FRONT), finds ORB feature matches, and triangulates them into 3D points.

Critical Detail: It forces a high feature count (nfeatures=20000) to ensure points are detected at the edges of the frame, which is the only place the side camera can see.

matcher.py (The Bridge)

Role: Connects the 3D world to the 2D Target Camera.

Key Function: project_points(points_3d, time_offset).

This function accepts a time_offset. If you set offset=0.1s, it calculates where the car would be 100ms later and projects the points based on that future pose. This is how we simulate and solve for delays.

optimizer.py (The Solver)

Role: The mathematical brain.

Logic: It defines a Cost Function (Average Pixel Error) and uses scipy.optimize to slide the time_offset variable until the error is minimized.

B. The Application

run_sync.py (Main Executable)

Role: The master script that ties everything together.

Workflow:

Loads Data.

Builds the Map.

Corruption Step: Injects a fake 50ms delay into the Matcher.

Runs the Optimizer to recover that delay.

Reports the final recovered offset (e.g., Recovered: 50.0009ms).

C. Validation & Tests (src/test/)

Use these to verify each component works in isolation.

test_data_loader.py

What it tests: Can we read the dataset? Is the trajectory smooth?

Success looks like: A plot showing a smooth vehicle path and a calculated speed of ~5-15 m/s.

test_map_maker.py

What it tests: Can we generate 3D points?

Success looks like: An image showing red dots tracked on trees/signs, and a scatter plot showing a cluster of points.

validate_sync.py (The "Cost Landscape" Test)

What it tests: Is the synchronization problem mathematically solvable?

How it works: It manually sweeps the time offset from -100ms to +100ms and plots the error.

Success looks like: A sharp "V-Shape" graph with the minimum exactly at 0ms. This graph proves your algorithm is robust and precise.

1. How to Run the Pipeline

Step 1: Verify Prerequisites

Ensure your data is in CAMERACALIBAFR/data/v1.0-trainval and CAMERACALIBAFR/data/samples.

Step 2: Run the Main Sync Tool

python run_sync.py


Expected Output:

[Step 3] SIMULATION: Injecting 50.0ms error.
...
[Step 5] Results
True Artificial Delay: 50.0000 ms
Recovered Offset:      50.0009 ms
SUCCESS: Synchronization Recovered!


Step 3: Generate the Engineering Proof

python src/test/validate_sync.py


Expected Output:
A window will pop up showing the "Optimization Cost Landscape" graph (the V-Curve). This image is the primary proof of performance for reports.