# 🛠️ EdgeForge-Vision
> **An Open-Source Edge-AI Framework for Accelerated Industrial Inspection and Telemetry on NVIDIA Jetson.**

EdgeForge-Vision is a modular, high-speed edge computing framework designed for real-time factory automation, industrial sorting, and object telemetry. Powered by the **NVIDIA Jetson Orin Nano**, it bridges hardware-accelerated camera pipelines directly to embedded control systems and web-based HMI dashboards.

---

## 🏗️ Framework Architecture Layout
This system is engineered using a decoupled, multi-threaded pipeline, making the core automation loops completely agnostic to the underlying deep learning model weights.

### 1. Hardware System Map
* **Vision Sensor:** Generic USB Webcams / CSI Cameras streaming via native Linux V4L2 interfaces.
* **Edge Compute Platform:** NVIDIA Jetson Orin Nano Developer Kit (System-on-Module).
* **Automation Controls:** Physical warning indicators / relays coupled natively to Jetson GPIO configurations (Board Pin 7).

### 2. Software Pipeline Matrix
* **Ingestion Core:** Headless GStreamer script routing raw hardware frame buffers directly into internal memory arrays, eliminating graphical desktop display dependence.
* **Inference Pipeline:** NVIDIA TensorRT (FP16 half-precision quantization) accelerating custom architectures directly on CUDA cores.
* **Concurrency Controls:** Thread-isolated execution loops bound using safe synchronization wrappers (`threading.Lock()`) between continuous vision tasks and asynchronous network queries.
* **HMI Dashboard:** Light-weight asynchronous Flask REST API server streaming system data maps over a customized Chromium Kiosk dashboard environment on auto-boot.

---

## 📂 Repository Layout
```text
EdgeForge-Vision/
├── camera/               # Independent camera ingestion profiles (CSI/USB)
├── models/               # Target workspace folder for compiled TensorRT (.engine) binaries
├── static/               # UI stylesheets, dark-themed frameworks & assets
├── templates/            # HTML5 Industrial HMI Dashboard panel screens
├── ui.py                 # Core Asynchronous Backend Engine & Multi-thread Coordinator
├── .gitignore            # System-level file exclusions map
└── README.md             # System Deployment Operations Manual
