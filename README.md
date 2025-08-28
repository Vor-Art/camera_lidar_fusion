# ROS2 Camera-LiDAR Fusion

[![License](https://img.shields.io/badge/License-MIT--Clause-blue.svg)](https://opensource.org/licenses/MIT)
[![ROS2 Version](https://img.shields.io/badge/ROS-Humble-green)](https://docs.ros.org/en/humble/index.html)

A ROS2 package for calculating **intrinsic** and **extrinsic calibration** between camera and LiDAR sensors. This repository provides an intuitive workflow to fuse data from these sensors, enabling precise projection of LiDAR points into the camera frame and offering an efficient approach to sensor fusion.

## Visual Overview
| **Static Sensors** | **Moving Sensors** |
|---------------------|--------------------|
| ![Static Sensors](assets/static_sensors.gif) | ![Moving Sensors](assets/moving_sensors.gif) |

---

## Table of Contents
1. [Get Started](#get-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
2. [Usage](#usage)
   - [Node Overview](#node-overview)
   - [Workflow](#workflow)
   - [Running Nodes](#running-nodes)
3. [License](#license)

---

## Get Started

### Prerequisites

To run this package, ensure the following dependencies are installed:
- **Git**: For version control and repository management.
- **Docker**: To streamline the environment setup and execution.
- **NVIDIA Container Toolkit** (if using an NVIDIA GPU): For hardware acceleration.

### Quick start using Docker
This repository includes a pre-configured Docker setup for easy deployment. To build the Docker image:
1. Run the build script:
   ```bash
   docker compose build
   ```
   This will create a Docker image named `ros2_camera_lidar_fusion`.

2. Once built, Launch the visuzlization:
   ```bash
   xhost +
   docker compose up rviz2 -d 
   ```

3. Enter the :
   ```bash
   docker compose run --rm --remove-orphans ros2_camera_lidar_fusion
   ```

---

## Usage

### Node Overview
This package includes the following ROS2 nodes for camera and LiDAR calibration:

| **Node Name**           | **Description**                                                                                       | **Output**                                     |
|--------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `get_intrinsic_camera_calibration.py`  | Saves the intrinsic from camera_info to the file.                                                    | Camera intrinsic calibration file.            |
| `save_sensor_data.py`    | Records synchronized data from camera and LiDAR sensors.                                             | Sensor data file.                             |
| `extract_points.py`      | Allows manual selection of corresponding points between camera and LiDAR.                            | Corresponding points file.                    |
| `get_extrinsic_camera_calibration.py` | Computes the extrinsic calibration between camera and LiDAR sensors.                                | Extrinsic calibration file.                   |
| `lidar_camera_projection.py` | Projects LiDAR points into the camera frame using intrinsic and extrinsic calibration parameters. | Visualization of projected points.            |

### Workflow
Follow these steps to perform calibration and data fusion:

1. **Extract Intrinsic to the file\***  
   ```bash
   ros2 run ros2_camera_lidar_fusion get_intrinsic_camera_calibration
   ```

2. **Collect the data\***  
   ```bash
   ros2 run ros2_camera_lidar_fusion save_data # (press Enter to capture)
   ```

3. **Label points pairs**  
   ```bash
   ros2 run ros2_camera_lidar_fusion extract_points
   ```

4. **Extrinsic Calibration**  
   ```bash
   ros2 run ros2_camera_lidar_fusion get_extrinsic_camera_calibration
   ```

5. **LiDAR Projection**  
   ```bash
   ros2 run ros2_camera_lidar_fusion lidar_camera_projection
   ```
> \* you should run `ros2 bag play` in parallel

## Maintainer
This package is maintained by:

**Artem Voronov**  
Email: [Artem.Voronov@skoltech.ru](mailto:Artem.Voronov@skoltech.ru)
GitHub: [Vor-Art](https://github.com/Vor-Art)  

**Clemente Donoso**  
Email: [clemente.donosok@gmail.com](mailto:clemente.donosok@gmail.com)
GitHub: [CDonosoK](https://github.com/CDonosoK)  

---

## License
This project is licensed under the **MIT**. See the [LICENSE](LICENSE) file for details.

---
Contributions and feedback are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
