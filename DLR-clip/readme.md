## DLR-CLIP: Dual-Level Refinement for Few-Shot Vision-Language Adaptation

Dual-Level Refinement (DLR) with CLIP for Few-Shot Learning. This project implements advanced multi-module fusion techniques for improving CLIP performance on few-shot learning tasks through Cross-Modal Adapters (CMA), Gate Logit Refiner (GLR), and Smoothed Knowledge Distillation (SKD).



### Abstract

DLR-CLIP (Dual-Level Refinement for Few-Shot Vision-Language Adaptation) addresses inter-class confusion in CLIP logits through a comprehensive multi-module approach:

**Key Components:**
1. **Cross-Modal Adapter (CMA)**: Lightweight layer-wise adapters for both text and visual transformers that enable fine-tuning of CLIP representations for specific downstream tasks
2. **Gate Logit Refiner (GLR)**: Asymmetric gated logit refinement with channel recalibration blocks for intelligent fusion of CLIP and CMA features
3. **Smoothed Knowledge Distillation (SKD)**: Knowledge distillation mechanism to stabilize and improve training

With its powerful visual-language alignment capability, CLIP performs well in zero-shot and few-shot learning tasks. However, CLIP's logits suffer from serious inter-class confusion problems in downstream tasks. Our DLR-CLIP method effectively learns and eliminates inter-class confusion in logits through intelligent module fusion. Experimental results show significant improvements in classification performance, especially in few-shot scenarios (1-shot, 2-shot, 4-shot, 8-shot, and 16-shot learning).


### Method

![Overall architecture of our LDC](./result/vis/arch.png)


### Data

* Follow [./datasets/0_dataset.md](./datasets/0_dataset.md) to install ImageNet and other 10 datasets or [Tip-Adapter DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) and [FAR DATASET.md](https://github.com/WideStars/FAR/blob/main/doc/DATASET.md) .


### Pretraining Model

* Download Pretraining Model to `./model/clip`

```
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
```


## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Before running experiments, configure the following in `dlr_train.py`:

* **`DATA_ROOT`**: Root directory where datasets are stored (default: `/media/yang/49f29042-389a-46e0-b8b1-94439dc013a5/data`)
* **`MODEL_CACHE_DIR`**: Directory for cached CLIP models (default: `./model/clip`)
* **`LOG_ROOT`**: Directory for experiment logs (default: `./result/log`)
* **`IMAGENET_ROOT`**: Root directory for ImageNet datasets (environment variable: `IMAGENET_ROOT` or default: `/media/yang/Elements SE`)

### Training Configuration

Modify parameters in `Config10Dataset` or `ConfigImageDomainShift` classes:
- `shots`: Number of few-shot samples (1, 2, 4, 8, 16)
- `backbone`: CLIP backbone ("RN50", "RN101", "ViT-B/32", "ViT-B/16")
- `lr`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 16)
- `train_epoch`: Number of training epochs (default: 50)
- `use_cma`: Enable Cross-Modal Adapter (default: False)
- `use_glr`: Enable Gate Logit Refiner (default: True)
- `use_skd`: Enable Smoothed Knowledge Distillation (default: False)
- `fixed_alpha`: Fixed alpha for GLR gate (None for adaptive)
- `fixed_weight`: Fixed weight for dynamic logit fusion (None for dynamic)


## Train & Test

### Single Experiment
```bash
python dlr_train.py
```

### Batch Experiments

**10 Datasets Ablation Study (ImageNet, FGVC, Caltech101, etc.):**
```bash
bash run_Ablation_10Datasets.sh
```

**ImageNet Ablation Study:**
```bash
bash run_Ablation_ImageNet.sh
```

**Few-Shot Experiments (1, 2, 4, 8, 16 shots):**
```bash
bash run_ImageNet_shots_1_2_4_8_16.sh
```

**Out-of-Distribution (OOD) Testing:**
```bash
python run_ImageNet_OOD.py
```

### Custom Experiments

For custom configurations, edit `run_custom_experiment.py` and run:
```bash
python run_custom_experiment.py
```


## Result


#### ResNet50


```
imagenet [60.482, 61.254, 62.47200000000001, 64.44, 66.632]
fgvc [18.69186918691869, 21.54215421542154, 26.432643264326433, 39.363936393639364, 53.46534653465347]
caltech101 [89.73630831643003, 90.22312373225152, 90.8316430020284, 92.61663286004057, 93.63083164300204]
stanford_cars [56.970526053973394, 59.76868548687975, 64.81780873025743, 74.38129585872404, 81.20880487501555]
dtd [50.11820330969267, 54.66903073286053, 59.33806146572104, 65.83924349881796, 71.63120567375887]
eurosat [73.92592592592592, 74.64197530864197, 80.49382716049382, 87.8888888888889, 90.19753086419753]
oxford_flowers [79.98375964271214, 86.47990255785626, 91.55501421031262, 93.74746244417376, 96.38652050345108]
food101 [77.36963696369637, 77.36963696369637, 77.8052805280528, 78.97689768976898, 79.28052805280528]
oxford_pets [87.2172254020169, 87.18997001907877, 88.06214227309894, 89.09784682474789, 90.54238212046879]
sun397 [62.584382871536526, 64.54911838790932, 67.81863979848866, 70.63476070528966, 72.72040302267003]
ucf101 [66.82527094898228, 69.8123182659265, 74.12106793550093, 77.02881311128735, 82.26275442770287]
```

#### ViT-B/16

```
imagenet [69.538, 69.858, 71.042, 72.482, 73.88]
fgvc [27.57275727572757, 28.982898289828984, 31.95319531953195, 37.983798379837985, 47.88478847884788]
caltech101 [94.15821501014199, 94.56389452332658, 95.25354969574038, 95.86206896551724, 96.63286004056795]
stanford_cars [68.23778137047631, 70.74990672801891, 75.12747170749907, 79.69158064917299, 84.20594453426192]
dtd [58.333333333333336, 62.174940898345156, 66.43026004728132, 71.63120567375887, 77.24586288416076]
eurosat [78.49382716049382, 81.72839506172839, 86.37037037037038, 90.80246913580247, 92.1604938271605]
oxford_flowers [83.63784003248071, 89.24076329679252, 94.07226958993098, 95.98051157125457, 97.8481526593585]
food101 [85.88448844884489, 86.07260726072607, 86.7095709570957, 86.93729372937294, 87.36633663366337]
oxford_pets [91.25102207686018, 91.16925592804579, 91.82338511856092, 92.47751430907604, 93.34968656309621]
sun397 [67.98992443324937, 69.60705289672543, 73.09319899244332, 75.5617128463476, 76.91183879093198]
ucf101 [73.248744382765, 75.94501718213058, 79.75151995770553, 81.31112873380914, 84.77398889770024]
```


### Visualization

**Generate Figure 5 (t-SNE Visualization):**
```bash
python dlr_vis_figure5.py
```

**Generate Confusion Matrix:**
```bash
python confusion_matrix_fgvc.py
```

**Generate t-SNE Embeddings:**
```bash
python t-sne.py
```

**Figure 5 - Logits Distribution Comparison:**

![Figure5](./result/vis/Figure5.png)


## Core Modules

### Cross-Modal Adapter (CMA)
- **File**: `cma.py`
- **Purpose**: Lightweight layer-wise adapters for text and visual transformers
- **Features**: 
  - Selective adapter application (layers `adapter_start` to `adapter_end`)
  - Support for half-precision (float16) models
  - Residual connections with learnable scaling

### Gate Logit Refiner (GLR)
- **File**: `glr.py`
- **Purpose**: Asymmetric gated logit refinement with channel recalibration
- **Features**:
  - Asymmetric gate only observes CLIP features (prevents CMA noise)
  - Channel Recalibration Block (CR) for inter-class deconfusion
  - Residual correction with learnable scaling
  - Support for fixed or adaptive alpha values

### Smoothed Knowledge Distillation (SKD)
- **File**: `skd_distillation.py`
- **Purpose**: Knowledge distillation with temperature scaling and label smoothing
- **Features**:
  - Temperature-based probability softening
  - Alpha-based label smoothing
  - KL divergence-based loss computation

### CLIP Model Extensions
- **File**: `clip_dlr/model.py`
- **Purpose**: Extended CLIP model with multi-scale fusion and adapter integration
- **Features**:
  - Multi-scale feature fusion (scale 1-4)
  - Learnable fusion weights
  - Integration with CMA and GLR modules

## Datasets

10 evaluation datasets are supported:
- **ImageNet**: 1,000 classes (large-scale benchmark)
- **Caltech-101**: 100 object categories
- **DTD**: 47 describable texture classes
- **FGVC Aircraft**: 100 aircraft families
- **EuroSAT**: 10 land use classes
- **Food-101**: 101 food types
- **Oxford Flowers**: 102 flower species
- **Oxford Pets**: 37 pet categories
- **SUN397**: 397 scene categories
- **UCF101**: 101 action categories

For dataset setup, follow instructions in [./datasets/0_dataset.md](./datasets/0_dataset.md)

## Citation

### BiBTeX

```
@InProceedings{Li_2025_CVPR,
    author    = {Li, Shuo and Liu, Fang and Hao, Zehua and Wang, Xinyi and Li, Lingling and Liu, Xu and Chen, Puhua and Ma, Wenping},
    title     = {Logits DeConfusion with CLIP for Few-Shot Learning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {25411-25421}
}
```


## Project Structure

```
DLR-CLIP/
├── clip_dlr/                 # CLIP model extensions
│   ├── clip.py              # CLIP loading and preprocessing
│   ├── model.py             # Extended CLIP model with multi-scale fusion
│   └── simple_tokenizer.py  # CLIP tokenizer
├── datasets/                # Dataset implementations
│   ├── utils.py             # Dataset utilities and transforms
│   ├── caltech101.py
│   ├── dtd.py
│   ├── eurosat.py
│   ├── fgvc.py
│   ├── food101.py
│   ├── imagenet.py
│   ├── oxford_flowers.py
│   ├── oxford_pets.py
│   ├── stanford_cars.py
│   ├── sun397.py
│   └── ucf101.py
├── result/                  # Results and logs
│   ├── log/                 # Training logs
│   └── vis/                 # Visualization outputs
├── cma.py                   # Cross-Modal Adapter module
├── glr.py                   # Gate Logit Refiner module
├── skd_distillation.py      # Smoothed Knowledge Distillation
├── dlr_train.py             # Main training script
├── dlr_vis_figure5.py       # Visualization script
├── run_*.sh                 # Batch experiment scripts
├── run_custom_experiment.py # Custom experiment runner
└── requirements.txt         # Python dependencies
```

## Acknowledgement

This project builds upon excellent open-source work:
- [CLIP](https://github.com/openai/CLIP) - Foundational vision-language model
- [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) - Adapter-based few-shot learning
- [FAR](https://github.com/WideStars/FAR) - Feature adaptation and refinement
- [CoOp](https://github.com/KaiyangZhou/CoOp) - Context optimization for CLIP

Special thanks to the original authors and contributors for their groundbreaking work in few-shot vision-language learning.
