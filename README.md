# 🛠️ EdgeForge-Vision
> **An Open-Source Agile Edge-AI Framework for Accelerated Industrial Inspection, Real-Time Object Telemetry, and Asynchronous HMI Controls on NVIDIA Jetson.**

EdgeForge-Vision is a production-grade, high-speed edge computing vision framework engineered for high-throughput factory automation, predictive sorting pipelines, and localized edge telemetry. Powered natively by the **NVIDIA Jetson Orin Nano**, this framework bridges hardware-accelerated camera ingestion modules directly with physical automation relays (via GPIO) and real-time asynchronous web HMI dashboards.

---

## 🏗️ Core Architecture Blueprint

This architecture is strategically built on a **decoupled, multi-threaded pipeline design pattern**. The core framework engine execution loops are completely agnostic to specific deep learning model weights.

```mermaid
graph TD
    A[Raw Hardware Sensors: CSI / USB V4L2] -->|Buffer Streaming| B[Headless GStreamer Ingestion Core]
    B -->|Thread-Isolated Memory Locking| C[Shared Memory Array Frame Buffer]
    C -->|Asynchronous Read Pipeline| D[NVIDIA TensorRT Inference Engine - FP16]
    C -->|Asynchronous Stream Matrix| E[Flask Local Web Server HMI Dashboard]
    D -->|Telemetry Output Evaluation| F[Jetson GPIO Control Loop: Board Pin 7]
    F -->|Hardware Signal| G[Physical Relay / Industrial Sirens Interlocking]
1. Hardware Interface Layers
Vision Acquisition: High-speed CSI / Generic USB Webcams streaming directly via internal Linux V4L2 kernel abstraction channels.

Edge Compute Platform: NVIDIA Jetson Orin Nano Developer Kit utilizing specialized System-on-Module (SoM) shared-memory spaces.

Automation Interlocks: Low-latency physical alerting devices/relays natively bound to Jetson GPIO configurations (Board Pin 7).

2. Software Concurrency Infrastructure
Ingestion Core: Headless GStreamer script pipeline streaming hardware frame buffers straight into volatile memory structures, entirely bypassing desktop window management systems for maximized throughput.

Inference Pipeline: Highly optimized NVIDIA TensorRT execution context deploying FP16 half-precision quantization matrices directly into Jetson CUDA cores.

Concurrency Matrix: Absolute isolation between vision loops and web networking queries utilizing strict synchronization wrappers (threading.Lock()) to guarantee zero-race conditions across shared buffers.

HMI Server Node: Lightweight, asynchronous Flask REST server rendering industrial status over an auto-booting responsive dark-themed dashboard environment.

Model Weights & Runtime Pre-requisites Access
Production-grade deep learning model weights (.pt, .onnx, .engine) and target compiled installation wheels (.whl components around 164MB+) are isolated from the repository's main git tree configuration layer to ensure optimal repository scalability and lightning-fast source replication.

Asset Workspace Binaries: You can instantly down-link all validated custom weights targets and the exact Jetson-optimized PyTorch execution binary components natively via the repository's Releases Workspace / Tag v1.0.0.

Target Workspace Mapping: Downloaded assets must be mapped directly into your local workspace directory paths exactly as listed below before runtime verification:

models/best.pt / models/yolov8n.pt

models/best.engine / models/yolov8n.engine

models/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

Multi-Industrial Application Blueprint (Cross-Domain Adaptability)Because EdgeForge-Vision decouples hardware ingestion from neural network model parameters, it can be seamlessly redeployed across diverse industrial setups without altering the system's foundational multi-threaded pipeline logic.Industry SectorPrimary Telemetry TaskCustom Model TargetPhysical Automation Response (GPIO Pin 7)Agricultural AutomationReal-Time Coconut Grade / Defect Countingcoconut_weights.engineRejects under-sized or damaged husks via pneumatic sorting valves.Beverage PackagingAssembly Line Bottle Volumetric Filling Controlbottle_counter.engineDiverts unsealed or underfilled containers into safety quarantine bins.Pharma OperationsBlister Pack Defective Pill Capsule Trackingpill_anomaly.engineHalts delivery conveyors instantly and sounds warning sirens.Logistics & WarehousingWarehouse Package Parcel Sorting & Classificationpackage_type.engineTriggers a directional actuator mechanism to slide boxes into correct delivery chutes.

Step-by-Step Custom Industrial Adaptation Operations Guide
Follow this explicit systems engineering manual to retrain the underlying model layout and adapt the repository code node from its default state to any target automation context (e.g., swapping a baseline sorting module with a custom factory bottle-counting array).

Step 1: Collect & Train Your Custom Weights
Source and label a domain-specific dataset (e.g., tracking plastic bottles, glass containers, defect parameters) using any annotation ecosystem.

Train a custom YOLO network using your high-end compute cluster workspace environment:
pip install ultralytics
yolo task=detect mode=train model=yolov8n.pt data=your_industrial_dataset.yaml epochs=100 imgsz=640

Once training pipelines wrap up, extract the newly compiled PyTorch optimization weights binary file named best.pt from your output directories.

Step 2: Compile Target Weights to High-Speed TensorRT Engine
For production-grade deployment with low-latency execution matrices on the NVIDIA Jetson Orin Nano, you must compile your .pt target weights array to a localized hardware-accelerated TensorRT network structure.

Run this command inside your target Jetson system execution framework environment:
# This triggers quantization down to FP16 half-precision on native CUDA cores
./venv/bin/python3 -c "from ultralytics import YOLO; model = YOLO('best.pt'); model.export(format='engine', device=0, half=True)"
This process converts your baseline model and outputs an ultra-fast hardware-serialized binary asset file named best.engine.
Step 3: Deploy Weights Assets into Code Workspace Directories
Transfer both your newly generated best.pt and best.engine assets into your edge target computer machine workspace.

Drop them into the empty placeholder directory structure array layout inside your workspace at:
EdgeForge-Vision/models/
Step 4: Map Global System Ingestion Parameters
Open the core application configuration code module ui.py using any standard IDE, and modify the global runtime constants definitions block located at the very top of the script code layer:
# ==============================================================================
# INDUSTRIAL RUNTIME PARAMETERS MATRIX CONFIGURATION
# ==============================================================================
# 1. Swap model weights target references dynamically 
MODEL_PATH = "models/best.engine"  # Set to your newly compiled customized TensorRT binary

# 2. Update Class Name Array Identifiers mapping to match your custom neural network training indices
CLASS_NAMES = ["Bottle-Full", "Bottle-Empty", "Cap-Defect"]  

# 3. Define target classification bounds trigger levels for GPIO relay alerts
CRITICAL_ALERT_CLASS = "Cap-Defect" 
# ==============================================================================

Step 5: Execute and Monitor Core Infrastructure
Initialize hardware tracking execution states over direct machine terminals:
sudo ./venv/bin/python3 ui.py
Open any Chromium container targeting localized loops at http://localhost:5000 to stream real-time factory dashboard data points.
Repository Layout Architecture Map
EdgeForge-Vision/
├── camera/               # Thread-isolated camera ingestion modules (CSI/USB handlers)
├── models/               # Target local repository workspace for hardware-bound TensorRT (.engine) binaries
│   └── .gitkeep          # Framework infrastructure safety placeholder token file
├── static/               # HMI UI stylesheets, custom dark-themed industrial assets
│   └── .gitkeep          # UI component framework structure placeholder token file
├── templates/            # HTML5 responsive real-time factory HMI management dashboard screens
├── ui.py                 # Core Asynchronous Backend Engine & System Thread Lock Coordinator
├── .gitignore            # System-level binary/OS layout exclusion files mapping rules
└── README.md             # System Deployment Operations & Custom Adaptation Manual

Engineering Evolution Roadmap
[ ] Integrate multi-stream hardware camera orchestration natively scaling across NVIDIA DeepStream SDK blocks.

[ ] Embed industrial field bus industrial protocols layers (Modbus/TCP, MQTT brokers, and OPC-UA nodes) for standard PLC networks synchronization.

[ ] Expand automated batch reporting workflows creating localized structural compliance tracking PDF log outputs.




    
