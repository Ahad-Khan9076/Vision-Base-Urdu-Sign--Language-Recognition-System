# Spatial-Temporal Joint-Attention Transformer (ST-JAT) for Vision-Based Urdu Sign Language Recognition

---

## 🔬 Core Architectural Mechanics & Innovation

Recognizing sentence-level sign language from continuous video inputs presents severe structural challenges: temporal frame redundancy, minor muscular tremors, and spatial information noise from stationary joints (e.g., hips, shoulders, and torso). 

Legacy architectures lean heavily on Recurrent Neural Networks (LSTMs/GRUs), which suffer from memory bottlenecks across extended sequence timelines and force sequential data processing. Conversely, Vanilla Transformers track global dependencies in parallel but frequently overfit to background coordinate artifacts because their attention layers evaluate all joints with uniform structural weight.

The **Spatial-Temporal Joint-Attention Transformer (ST-JAT)** solves this paradigm by injecting localized downsampling layers prior to spatial matrix self-attention, processing high-dimensional coordinate arrays across five clear operations:

[Raw Input Sequences: .avi + .eaf File Elements]
│
▼
Stage 1: Multi-Modal Tracking Parsing (144 Spatial Landmarks)
│
▼
Stage 2: Linear Hidden Vector Projection ($hidden\_dim = 256$)
│
▼
Stage 3: Multi-Tier Temporal Encoder (Conv1D Timeline Compression)
│  └─ Halves continuous frame lengths from 60 down to 30
▼
Stage 4: Softmax-Filtered Joint Attention Layer (Spatial Masking)
│  └─ Parallel Queries, Keys, Values Matrix Multiplication
▼
Stage 5: Stacked Transformer Encoders Block ($N_x = 2$, 4 Attention Heads)
│
▼
Stage 6: Regularized GELU Classifier Head (Global Average Pooling ──> Output)

### 1. Geometric Vectorization
Continuous video matrices ($X_{\text{raw}} \in \mathbb{R}^{B \times T \times C}$) are extracted frame-by-frame via a localized computer vision pipeline. The upper body skeleton provides 11 tracking points ($11 \times [x, y] = 22$), while both hands provide 21 localized joints each ($21 \times 3 \times 2 = 126$), combining to form a robust **144-dimensional feature vector per frame**.

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

## 📂 Repository Layout

```text
.
├── dataset_collection/
│   └── dataset_collection_code.py       # Code base handling raw camera captures
├── dataset_preprocessing/
│   ├── dataset_scanning.py             # Directory discovery loop for video files
│   └── process_features.py              # Geometric keypoint pipeline pipeline logic
├── dataset_split/
│   ├── create_split_index.py            # Generates relational track arrays split entries
│   ├── initial_scan_manifest.csv        # Log trace output recording original scans
│   ├── master_dataset_index.csv         # Consolidated dataset index map
│   └── sentence_mapping.csv             # Index string mapping labels dictionary
├── experimentations/
│   ├── train_baseline_lstm.py           # Benchmark implementation for standard LSTM
│   ├── train_diagram_architecture.py    # Main ST-JAT production training loop execution
│   ├── train_hybrid_cnnlstm.py          # Spatial-temporal hybrid baseline network pipeline
│   └── train_transformer_ultra.py       # Vanilla global transformer benchmark setup
├── FYPapp.zip                           # Packaged desktop/inference presentation app
├── collect_sign_dataset.py              # Independent feature extraction run environment
├── README.md                            # Comprehensive system documentation
└── requirements.txt                     # Complete software package requirement definitions

📊 Empirical Performance & Benchmark SuiteThe ST-JAT network was evaluated under identical training baselines (120 Epochs, Batch Size 64, Cosine Annealing Learning Rate Scheduler) against traditional sequential frameworks:Architecture ProfileParameter FootprintPeak Val AccuracyGlobal Min Val LossInference LatencyConvergence EpochBaseline LSTM38 MB96.97%0.131142.1 ms~88 (Slow tracking)CNN-LSTM Hybrid54 MB98.48%0.084328.4 ms~65 (Volatile boundary)Vanilla Transformer24 MB99.49%0.045518.2 ms~42 (Noise sensitive)Proposed ST-JAT16 MB99.75%0.030911.5 ms< 35 (Ultra-fast)
