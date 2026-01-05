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


```
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




### Custom Experiments

For custom configurations, edit `run_custom_experiment.py` and run:
```bash
python run_custom_experiment.py
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
