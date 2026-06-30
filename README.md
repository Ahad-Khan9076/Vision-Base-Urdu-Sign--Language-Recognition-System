# Spatial-Temporal Joint-Attention Transformer (ST-JAT) for Vision-Based Urdu Sign Language Recognition

---

## 🔬 Core Architectural Mechanics & Innovation

Recognizing sentence-level sign language from video inputs presents unique challenges: temporal frame redundancy, tracking noise from minor tremors, and spatial contamination from stationary joints (like hips or shoulders). 

Legacy approaches rely on Recurrent Neural Networks (LSTMs/GRUs), which suffer from memory bottlenecks over long sequences and process frames one by one, creating significant latency. Conversely, Vanilla Transformers capture global context in parallel but often overfit to background noise because their attention mechanisms treat all joints equally.

The **Spatial-Temporal Joint-Attention Transformer (ST-JAT)** addresses these challenges through a specialized architecture that processes spatial-temporal vectors in five distinct stages:

### 1. Geometric Vectorization
Continuous video matrices ($X_{\text{raw}} \in \mathbb{R}^{B \times T \times C}$) are extracted frame-by-frame via a specialized computer vision pipeline. The upper body skeleton provides 11 tracking points ($11 \times [x, y] = 22$), while both hands provide 21 localized joints each ($21 \times 3 \times 2 = 126$), combining to form a robust **144-dimensional feature vector per frame**.

### 2. Multi-Tier Temporal Encoder Block (Conv1D Downsampling)
Before computing global dependencies, the tensor is permuted to a channel-first layout ($B \times 256 \times 60$) and routed through a dual-pathway Convolutional Network:
* **Main Pathway:** Captures immediate frame-to-frame shifts: $\text{Dropout}_{0.1}(\text{BatchNorm1D}(\text{ReLU}(\text{Conv1D}(H_0))))$.
* **Auxiliary Pathway:** Compresses the time dimension using a 1D Max Pooling layer with a stride of 2, reducing the sequence from **60 frames to 30 states**. This optimization cuts downstream self-attention computational complexity from $O(60^2)$ to $O(30^2)$, minimizing latency during deployment.

### 3. Softmax-Filtered Joint Attention Layer
To prevent the model from tracking stationary joints, the sequence is mapped into Query ($Q$), Key ($K$), and Value ($V$) spaces. A spatial attention mask isolates active movements using scaled dot-products and an explicit residual identity mapping:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
$$\text{Output}_{\text{Attention}} = \text{Softmax}\left(W_o(\text{Attention}(Q,K,V)) + V\right)$$

### 4. Stacked Transformer Encoders
The isolated 30-step sequence is passed through a deep multi-head self-attention module ($N_x=2$) featuring 4 parallel attention heads and a feed-forward expansion size of 512. This allows the model to analyze global dependencies across the entire timeline simultaneously.

### 5. Regularized Classifier Head
The final hidden matrices are condensed into a 256-dimensional vector using Global Average Pooling (GAP). This representation passes through dense layers utilizing **GELU (Gaussian Error Linear Unit)** activations and a 30% dropout regularizer to generate raw multi-class probability scores across the 20 target sentence classes.

---

## 📊 Empirical Performance & Benchmark Suite

The ST-JAT network was evaluated under identical training baselines (**120 Epochs**, **Batch Size 64**, **Cosine Annealing Learning Rate Scheduler**) against traditional sequential frameworks:

| Architecture Profile | Parameter Footprint | Peak Val Accuracy | Global Min Val Loss | Inference Latency | Convergence Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM** | 38 MB | 96.97% | 0.1311 | 42.1 ms | ~88 (Slow tracking) |
| **CNN-LSTM Hybrid** | 54 MB | 98.48% | 0.0843 | 28.4 ms | ~65 (Volatile boundary) |
| **Vanilla Transformer**| 24 MB | 99.49% | 0.0455 | 18.2 ms | ~42 (Noise sensitive) |
| **Proposed ST-JAT** | **16 MB** | **99.75%** | **0.0309** | **11.5 ms** | **< 35 (Ultra-fast)** |

---

## 📂 Repository Layout

├── build_eaf_avi_pkl.py         # Recursive preprocessing pipeline for .eaf/.avi files \n
├── model.py                    # Modular ST-JAT PyTorch architecture definition
├── train.py                    # Production training loop with Cosine Annealing
├── requirements.txt            # Python ecosystem dependency definitions
└── README.md                   # Complete system documentation


---

## 🛠️ Local Implementation Quickstart

### 1. Environment Activation & Initialization
Isolate your packages within your active local virtual environment:

# Activate the Urdu Sign Language environment
source /home/ahad/Documents/fyp_dataset/env_urdu_sl/bin/activate

# Install core tensor and linguistic parsing packages
pip install pympi-ling opencv-python scikit-learn tqdm numpy pandas torch torchinfo cvzone

2. High-Speed Binary Serialization (.pkl)
Compile raw multi-modal datasets (.avi videos + ELAN .eaf files) across all 20 sentence subfolders into highly optimized training blocks:
python3 /home/ahad/Documents/fyp_dataset/build_eaf_avi_pkl.py

This script balances data splits (80% Train, 20% Val), standardizes lengths, and outputs binary files directly to /home/ahad/Documents/fyp_pipeline_outputs/fyp_processed_dataset/.

