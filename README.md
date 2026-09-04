# Swarm-Based Knowledge Transfer for Heterogeneous ML Models

## Overview

This project presents a **Swarm-Intelligent Knowledge Transfer Framework for Heterogeneous Machine Learning Models**.

The framework enables multiple machine learning models with different architectures and capabilities to collaborate during inference through a decentralized knowledge-sharing mechanism.

The system employs three heterogeneous deep learning agents:

* **MobileNet-V3 Small** – Lightweight perception agent
* **EfficientNet-B0** – Medium-complexity perception agent
* **ResNet-18** – High-capability perception agent

Instead of exchanging raw video data, training datasets, model parameters, or feature maps, the agents communicate using lightweight **Knowledge Packets**.

A **Collective Memory Module** selects the most reliable knowledge based on prediction confidence and shares it across the swarm.

---

## Problem Statement

Traditional machine learning systems often operate as independent models or rely on homogeneous architectures. Different models have varying computational capabilities, feature representations, inference speeds, and prediction performance.

The challenge is to enable heterogeneous machine learning models to collaborate and share useful knowledge without requiring:

* Exchange of raw input data
* Exchange of model parameters
* Retraining of participating models
* Identical model architectures

This project addresses this challenge using a swarm-intelligent knowledge-sharing framework that enables autonomous agents to exchange compact semantic knowledge during inference.

---

## Proposed Framework

The proposed system processes an input video and divides it into individual frames.

Each frame is independently processed by three heterogeneous machine learning agents:

1. **MobileNet-V3 Small**
2. **EfficientNet-B0**
3. **ResNet-18**

Each agent generates:

* Object label
* Confidence score

The inference output is converted into a standardized **Knowledge Packet**.

All packets are sent to the **Collective Memory Module**, where the packet with the highest confidence is selected as the most reliable knowledge representation.

The selected knowledge is then shared with the participating agents, enabling collaborative perception and knowledge reuse across heterogeneous models.

---

## System Workflow

```text
Input Video
     │
     ▼
Frame Extraction
     │
     ▼
┌─────────────────────────────────────┐
│   Heterogeneous Perception Agents   │
│                                     │
│  MobileNet │ EfficientNet │ ResNet  │
└─────────────────────────────────────┘
     │
     ▼
Local Predictions + Confidence Scores
     │
     ▼
Knowledge Packet Generation
     │
     ▼
Collective Memory
     │
     ▼
Confidence-Based Knowledge Selection
     │
     ▼
Best Knowledge Packet
     │
     ▼
Knowledge Sharing and Transfer
     │
     ▼
Knowledge Integration
     │
     ▼
Performance Evaluation
```

---

## Knowledge Packet

After processing each frame, every agent generates a structured knowledge packet.

Each packet contains:

| Field            | Description                                |
| ---------------- | ------------------------------------------ |
| Frame ID         | Identifier of the processed frame          |
| Object Label     | Predicted class, such as Human / No Human  |
| Confidence Score | Confidence associated with the prediction  |
| Timestamp        | Time at which the prediction was generated |
| Agent Identifier | Identifier of the source model             |

The standardized packet enables **model-independent communication** among heterogeneous machine learning agents.

---

## Collective Memory

The **Collective Memory Module** acts as a shared knowledge repository for the swarm.

For each processed frame:

1. Knowledge packets are received from all agents.
2. The confidence scores are compared.
3. The packet with the highest confidence is selected.
4. The selected packet is stored as the best knowledge representation.
5. The best knowledge is shared with other participating agents.

This approach reduces redundant storage and allows the swarm to retain the most reliable perception knowledge.

---

## Mathematical Representation

Let the swarm contain \(N\) heterogeneous agents:

$$
S = \{A_1, A_2, \dots, A_N\}
$$

For an input frame \(F_t\), each agent produces a prediction:

$$
P_i(F_t) = (L_i, C_i)
$$

where:

* \(L_i\) is the predicted object label.
* \(C_i\) is the confidence score.
* \(A_i\) is the corresponding ML agent.

Each prediction is represented as a knowledge packet:

$$
K_i = \{F_t, L_i, C_i, T_i, A_i\}
$$

The Collective Memory selects the best packet using confidence-based selection:

$$
K_{best} = \arg\max_{K_i} C_i
$$

For the three-agent implementation:

$$
C_{best} = \max(C_{MobileNet}, C_{EfficientNet}, C_{ResNet})
$$

The selected knowledge packet is stored in collective memory and shared across the swarm.

---

## Models Used

### MobileNet-V3 Small

MobileNet-V3 Small serves as the lightweight perception agent.

**Role in the swarm:**

* Lightweight inference
* Low computational cost
* Real-time perception
* Enables resource-constrained agents to participate in the swarm

### EfficientNet-B0

EfficientNet-B0 acts as a medium-complexity perception agent.

**Role in the swarm:**

* Balanced computational efficiency
* Strong prediction capability
* Contributes perception knowledge to the collective memory

### ResNet-18

ResNet-18 serves as a high-capability perception agent with strong feature representation.

**Role in the swarm:**

* Reliable perception
* Strong feature extraction
* High-quality knowledge contribution

---

## Functional Modules

### 1. Video Acquisition and Frame Extraction

* Loads the input video
* Extracts individual frames
* Preprocesses frames
* Supplies frames to all perception agents

### 2. Heterogeneous Perception Agents

The framework uses:

* MobileNet-V3 Small
* EfficientNet-B0
* ResNet-18

Each agent independently performs inference on the same frame.

### 3. Knowledge Packet Generation

Converts model predictions into standardized semantic knowledge packets.

### 4. Collective Memory

* Stores knowledge packets
* Compares confidence scores
* Selects the best packet
* Maintains shared swarm knowledge

### 5. Knowledge Sharing

Distributes the selected best knowledge packet among participating agents.

### 6. Knowledge Integration

Allows receiving agents to utilize shared knowledge and reinforce their local predictions.

### 7. Performance Evaluation

Evaluates the system by comparing performance before and after knowledge transfer.

---

## Technologies Used

| Technology  | Purpose                               |
| ----------- | ------------------------------------- |
| Python 3.x  | Primary programming language          |
| PyTorch     | Deep learning framework               |
| TorchVision | Pre-trained model support             |
| OpenCV      | Video processing and frame extraction |
| NumPy       | Numerical computation                 |
| Pandas      | Data processing and analysis          |
| Matplotlib  | Result visualization                  |
| Seaborn     | Performance visualization             |
| Fakeredis   | Collective memory simulation          |
| JSON        | Knowledge packet storage and exchange |
| VS Code     | Development environment               |
| Git         | Version control                       |

---

## Project Structure

```text
ASwarmIntelligentFrameworkForHeterogeneousMLModels/
│
├── CODE/
│   └── Swarm/
│       └── Swarm/
│           ├── data/
│           ├── demo_runs/
│           ├── docs/
│           ├── tools/
│           ├── paper_results_*/
│           └── source files
│
├── IMAGES/
│
├── Literature Survey/
│
├── Report/
│
├── Existing vs proposed.png
├── Flowchart.png
├── Mathematical Demonstration.png
├── RESULTS TABLE.png
├── System Architecture.jpeg
├── System Description.jpeg
│
├── .gitignore
└── README.md
```

> Duplicate project copies and local cache files are excluded from version control.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Zhussain13/ASwarmIntelligentFrameworkForHeterogeneousMLModels.git
```

### 2. Navigate to the Project

```bash
cd ASwarmIntelligentFrameworkForHeterogeneousMLModels
```

### 3. Navigate to the Source Code

```bash
cd CODE/Swarm/Swarm
```

### 4. Create a Virtual Environment

```bash
python -m venv venv
```

### 5. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### 6. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib seaborn fakeredis
```

---

## Usage

The general execution workflow is:

1. Select or provide an input video.
2. Extract frames from the video.
3. Initialize the three heterogeneous ML agents.
4. Perform local inference for each frame.
5. Generate knowledge packets.
6. Send packets to collective memory.
7. Select the highest-confidence knowledge packet.
8. Share the selected knowledge with other agents.
9. Integrate shared knowledge.
10. Evaluate performance before and after knowledge transfer.

Refer to the source code and documentation inside the `CODE/Swarm/Swarm` directory for the available execution scripts and experimental configurations.

---

## Performance Evaluation

The framework evaluates the impact of swarm-based knowledge transfer by comparing model performance:

* Before knowledge transfer
* After knowledge transfer

The evaluation focuses on:

* Prediction accuracy
* Confidence improvement
* Collaborative perception
* Knowledge reuse
* Reduction in redundant inference
* Efficient knowledge storage
* Overall swarm intelligence

Experimental artifacts and generated results are included in the project directories.

---

## Key Contributions

* Swarm-intelligent framework for heterogeneous machine learning models.
* Knowledge transfer during inference without model retraining.
* Model-independent communication using standardized knowledge packets.
* Confidence-based selection of reliable perception knowledge.
* Collective memory for efficient knowledge storage and reuse.
* Collaboration between lightweight and high-capability ML models.
* Reduced communication overhead by avoiding raw data and parameter exchange.

---

## Research Objective

The primary objective of this work is to investigate whether swarm intelligence can enable effective knowledge sharing and collaborative perception among heterogeneous machine learning models.

The framework aims to demonstrate that:

> **Collective knowledge sharing can help heterogeneous ML agents benefit from the strengths of one another without exchanging model parameters or raw data.**

---

## Documentation

This repository includes:

* Project reports
* Literature survey
* System architecture
* System description
* Flowchart
* Mathematical demonstration
* Experimental results
* Performance comparison artifacts

---

## Author

**Shaik Mohammed Zaheed Hussain**

---

## Academic Information

**Department of Computer Science and Engineering**
**CMR Institute of Technology, Bengaluru**

**Year:** 2026

---

## License

This project is intended for **academic and research purposes**.

Please contact the author for reuse or redistribution beyond academic use.
