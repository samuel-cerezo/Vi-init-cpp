# 📌 Visual-Inertial Initialization using Small-Angle Approximation

This repository provides a lightweight and modular C++ implementation of a **fully closed-form visual-inertial initialization algorithm**, based on the small rotation approximation.

## ✨ Features

- Closed-form gyroscope bias estimation from two-view visual correspondences and IMU preintegration
- Linear estimation gyroscope bias
- Optional refinement via nonlinear optimization (Ceres Solver)
- Real-time performance
- Modular, testable architecture with full support for EuRoC dataset

## 🧱 Dependencies

Make sure the following libraries are installed:

- **C++17 compiler**
- [Eigen3](https://eigen.tuxfamily.org/) (e.g., `brew install eigen`)
- [OpenCV >= 4](https://opencv.org/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [pybind11](https://github.com/pybind/pybind11)
- [Ceres Solver](http://ceres-solver.org/)
- Python 3.8 (or 3.9) with the [C2P module](https://github.com/javrtg/C2P) installed

## 📂 Project Structure

```
ViInitCpp/
├── CMakeLists.txt
├── include/
│   ├── camera_utils.h
│   ├── feature_tracking.h
│   ├── frame_processor.h
│   ├── euroc_io.h
│   ├── c2p_wrapper.h
│   └── lie_utils.h
├── src/
│   ├── main.cpp
│   ├── camera_utils.cpp
│   ├── feature_tracking.cpp
│   ├── frame_processor.cpp
│   ├── euroc_io.cpp
│   ├── c2p_wrapper.cpp
│   ├── bg_small_angle.cpp
│   └── bg_optimization.cpp
├── data/
│   └── MH_02_easy/   # EuRoC dataset (you must download separately)
├── results.csv       # Output file with per-frame errors and timings
```

## ⚙️ Build Instructions

```bash
# Clone and enter the repository
git clone https://github.com/your_username/vi-init-cpp.git
cd vi-init-cpp

# Create and enter build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j
```

> 🧠 Tip: You may need to set OpenCV, Python and yaml-cpp paths manually in `CMakeLists.txt` if using Apple Silicon or custom installations.

## ▶️ Run the Algorithm

Before running, download the **EuRoC dataset** (e.g., MH_02_easy) and place it under `data/`.

```bash
./vi_init
```

The output will be printed in the terminal and stored in `results.csv`, including:

- Frame index
- Error of the closed-form gyroscope bias
- Optimization error (optional)
- Time taken (microseconds)

## 🧪 Example Results (MH_02_easy)

| Method                 | Mean Error (rad/s) | Time [μs] |
|------------------------|--------------------|-----------|
| Closed-form (Eq. 12)   | 0.017              | 32        |
| Optimization (Ceres)   | 0.011              | 3713      |

## 📚 Reference

This implementation accompanies the article:

> **Decoupled Visual-Inertial State Initialization with the Small Rotation Approximation**  
> *Samuel Cerezo, Javier Civera*  
> Submitted to IEEE Robotics and Automation Letters, 2025

If you use this code, please cite the paper when available.

## 🔧 TODO

- Add multi-frame linear estimation for full state
- Integrate with VIO backend (e.g., ORB-SLAM3, VINS-Mono)
- Add plotting script for `results.csv`

## 📬 Contact

For questions, feel free to open an issue or contact:

- Samuel Cerezo — [samucerezo@domain.com](mailto:samucerezo@domain.com)
