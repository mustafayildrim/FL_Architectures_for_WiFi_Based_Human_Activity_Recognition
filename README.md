# FL Architectures for WiFi-Based Human Activity Recognition

A course project for **ECE1508** that implements and compares federated learning architectures for **WiFi-based Human Activity Recognition (HAR)** using **Channel State Information (CSI)** data.

---

## Overview

This project compares two federated learning architectures:

- **Centralized Federated Learning (FedAvg)**
- **Decentralized Federated Learning (Graph-Based FL with Ring Topology)**

using the **UT-HAR** dataset and three deep learning models.

The project is motivated by the lack of controlled comparisons between centralized and decentralized FL for CSI-based HAR. The main goal is to evaluate the **accuracy-communication trade-off** across architectures under a consistent experimental setup.

---

## Dataset

**UT-HAR** is a WiFi CSI-based human activity recognition dataset.

| Property | Value |
|---|---|
| Input shape | 1 × 250 × 90 |
| Training samples | 3,977 |
| Validation samples | 496 |
| Test samples | 500 |
| Number of classes | 7 |

**Activity Classes:**  
Lying Down, Falling, Walking, Running, Pickup, Sitting Down, Standing Up

**Dataset notes:**
- The class distribution is **non-uniform**
- CSI inputs are represented as matrices with shape **1 × 250 × 90**

---

## Data Preprocessing

The preprocessing pipeline includes:

- **Normalization:** Min-Max normalization to \([0,1]\)
- **Augmentation:** Gaussian Noise, Temporal Shift, and Random Scaling
- **Class imbalance handling:** `WeightedRandomSampler`
- **Data partitioning:** IID split across \(K\) clients using random shuffling

---

## FL Architectures

### Centralized FL (FedAvg)

FedAvg uses a central server. Each client performs local training, sends model updates to the server, and the server aggregates them through averaging.

### Decentralized FL (Graph-Based FL, Ring Topology)

The decentralized setup removes the central server. Each client communicates only with its two neighboring clients in a ring and performs local aggregation through weighted averaging.

---

## Models

| Model | Parameters | Notes |
|---|---|---|
| Custom CNN | 2.82M | Lightweight and purpose-built for CSI matrices |
| DenseNet121 | 6.95M | Dense connections with strong feature reuse |
| ResNet50 | 23.52M | Deep residual network |

**Model setup:**
- All models are trained **from scratch**
- The input layer is adapted from **3-channel** input to **1-channel** input

---

## Hyperparameter Tuning

A grid search is performed over:

- **Number of clients:** $K \in \{5, 10\}$
- **Learning rate:** $lr \in \{0.001, 0.0005, 0.0001\}$

**Fixed settings:**
- Optimizer: Adam
- Global rounds: $T = 10$
- Local epochs: $E = 5$
- Batch size: 32

**Model selection criterion:** final-round validation accuracy

The same tuning protocol is applied to both **FedAvg** and **Graph-Based FL**.

---

## Results

### Centralized FL (FedAvg)

#### Best Test Performance by Model

| Model | K | lr | Test Accuracy | Test F1 |
|---|---|---|---|---|
| CNN | 5 | 0.001 | 95.60% | 0.9329 |
| ResNet50 | 5 | 0.0005 | 97.20% | 0.9671 |
| DenseNet121 | 5 | 0.0001 | **98.20%** | **0.9700** |

#### Main Findings
- A **smaller number of clients** achieved higher accuracy because each client retained more local data after partitioning
- \(K=10\) **doubles communication cost** but usually did **not** improve accuracy
- **Smaller learning rates** worked best for **ResNet50** and **DenseNet121**
- The **CNN** performed best at **\(lr=0.001\)**
- The best overall FedAvg model was **DenseNet121** with **98.20% test accuracy** at **\(K=5\)** and **\(lr=0.0001\)**

---

### Decentralized FL (Graph-Based, Ring-Based)

#### Best Test Performance by Model

| Model | K | lr | Test Accuracy | Test F1 |
|---|---|---|---|---|
| CNN | 5 | 0.001 | 91.72% | 0.8947 |
| ResNet50 | 5 | 0.0001 | 91.24% | 0.8870 |
| DenseNet121 | 5 | 0.0001 | **93.40%** | **0.9164** |

#### Main Findings
- Graph-Based FL produced **less accurate models** than FedAvg, with the best result reaching **93.40%**
- Training showed **more fluctuations and instabilities**
- Convergence was **slower** because information propagated only through local ring neighbors
- Deep models provided roughly **1–5% higher accuracy**
- Communication cost could be much higher, reaching about **10× more** in some settings
- Simpler CNNs may still be attractive for **real-world applications** when efficiency matters

---

## Overall Comparison

- **FedAvg consistently outperformed Graph-Based FL** across all three models
- **DenseNet121** achieved the best performance in both architectures
- **\(K=5\)** outperformed **\(K=10\)** in both settings
- The decentralized ring topology introduced slower information flow, which reduced accuracy and training stability
- The project highlights a clear **accuracy-communication trade-off** between centralized and decentralized FL for WiFi-based HAR

---

## Dependencies

```bash
torch
torchvision
numpy
matplotlib
scikit-learn
networkx
