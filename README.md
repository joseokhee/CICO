# CICO — Supervised Contrastive Learning with Importance-based CutOut for Few-shot Image Classification

Official code for

> **Supervised Contrastive Learning with Importance-based CutOut for Few-shot Image Classification**<br>
> Seokhee Jo, Jungmin So — *ACM SAC 2025* (oral)

Built on top of the **P>M>F** pipeline ([hushell/pmf_cvpr22](https://github.com/hushell/pmf_cvpr22), CVPR 2022). The upstream git history is kept in this repository; my changes are in the commits on top of it.

## Method

During episodic meta-training on a ViT backbone (pre-trained with DINO):

1. **Importance-based CutOut** — For every support and query image, the backbone's last-layer self-attention from the `[CLS]` token ranks the patches by importance. The **most attended patches are cut out** (masked to zero) and the masked images are used for training, so the model cannot rely only on the single most discriminative region and has to learn from the rest of the object.
   → `attention_image()` in [`engine.py`](engine.py), using `get_last_selfattention()` exposed in [`models/protonet.py`](models/protonet.py).

2. **Supervised contrastive objective** — Prototype similarities are optimized with a per-class binary cross-entropy against one-hot labels instead of softmax cross-entropy, so each query is pulled toward its own class prototype and pushed away from the others independently.
   → `criterion = BCELoss()` in [`main.py`](main.py), training loop in [`engine.py`](engine.py).

## Changes from P>M>F

| File | Change |
|---|---|
| `engine.py` | Importance-based CutOut on support/query images; one-hot BCE objective; train accuracy logging |
| `models/protonet.py` | Expose last self-attention of the backbone |
| `main.py` | Loss switched to BCE; separate log / checkpoint names |
| `datasets/tiered_imagenet.py`, `datasets/__init__.py`, `utils/args.py` | tiered-ImageNet support |
| `datasets/cdfsl/*` | Cross-domain evaluation (ChestX, ISIC, CropDisease) adjustments |

## Usage

Environment, dataset preparation, and commands follow P>M>F — see [the upstream README](https://github.com/hushell/pmf_cvpr22#prerequisites).

```bash
pip install -r requirements.txt

# meta-train on mini-ImageNet (5-way 5-shot) with a DINO ViT-S/16 backbone
python main.py --output_dir outputs/cico_mini --dataset mini_imagenet \
    --arch dino_small_patch16 --nSupport 5 --epochs 100
```

Supported datasets: CIFAR-FS, mini-ImageNet, tiered-ImageNet, Meta-Dataset; cross-domain evaluation on CDFSL via `test_bscdfsl.py`.

## Citation

If you use this code, please also cite P>M>F:

```bibtex
@inproceedings{hu2022pmf,
  author    = {Hu, Shell Xu and Li, Da and St\"uhmer, Jan and Kim, Minyoung and Hospedales, Timothy M.},
  title     = {Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference},
  booktitle = {CVPR},
  year      = {2022}
}
```
