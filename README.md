# CICO — Supervised Contrastive Learning with Importance-based CutOut for Few-shot Image Classification

Official code for

> **Supervised Contrastive Learning with Importance-based CutOut for Few-shot Image Classification**<br>
> Seokhee Jo, Jungmin So — *ACM SAC 2025* (oral)

Built on top of the **P>M>F** pipeline ([hushell/pmf_cvpr22](https://github.com/hushell/pmf_cvpr22), CVPR 2022). The upstream git history is kept in this repository; my changes are in the commits on top of it.

## Method

During episodic meta-training on a ViT backbone (pre-trained with DINO):

1. **Importance-based CutOut** — For every support and query image, the backbone's last-layer self-attention from the `[CLS]` token ranks the patches by importance. The **most attended patches (top-k, `k=10` in `attention_image()`) are cut out** (masked to zero) and the masked images are used for training, so the model cannot rely only on the single most discriminative region and has to learn from the rest of the object.
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

### Evaluation

Checkpoints and logs are written to `--output_dir` as `wo_cico_checkpoint.pth`, `wo_cico_best.pth` and `log_wo_cico.txt`.

```bash
# in-domain test (mini-ImageNet / CIFAR-FS / tiered-ImageNet): same arguments as training, plus --eval and --resume
python main.py --output_dir outputs/cico_mini --dataset mini_imagenet \
    --arch dino_small_patch16 --nSupport 5 \
    --eval --resume outputs/cico_mini/wo_cico_best.pth

# cross-domain test on CDFSL (set --n_shot to 5, 20 or 50)
python test_bscdfsl.py --test_n_way 5 --n_shot 5 --device cuda:0 \
    --arch dino_small_patch16 --deploy vanilla \
    --output_dir outputs/cico_mini --resume outputs/cico_mini/wo_cico_best.pth

# Meta-Dataset (single GPU; see the P>M>F README for the DDP / fine-tuning variants)
python test_meta_dataset.py --data-path /path/to/meta_dataset/ --dataset meta_dataset \
    --arch dino_base_patch16 --deploy vanilla --device cuda:0 \
    --output_dir outputs/cico_md --resume outputs/cico_md/wo_cico_best.pth
```

## Results

Results reported in the paper (5-way accuracy, %, averaged over 2,000 test episodes). This repository contains the ViT / P>M>F version of CICO; the paper's CNN results (SimpleShot, PT-MAP with Grad-CAM based CutOut) are not included here.

**mini-ImageNet and CIFAR-FS**

| Method | Backbone | CIFAR-FS 1-shot | CIFAR-FS 5-shot | mini-ImageNet 1-shot | mini-ImageNet 5-shot |
|---|---|---|---|---|---|
| P>M>F | ViT-S/16 | 81.1 | 92.5 | 93.1 | 98.0 |
| **P>M>F + CICO** | ViT-S/16 | **83.8** | **92.9** | **93.8** | **98.2** |
| P>M>F | ViT-B/16 | 84.3 | 92.2 | 95.3 | 98.4 |
| **P>M>F + CICO** | ViT-B/16 | **86.2** | **93.8** | **95.7** | **98.7** |

**Ablation on CIFAR-FS** (loss × Importance-based CutOut)

| Backbone | Loss | CutOut | 1-shot | 5-shot |
|---|---|---|---|---|
| ViT-S/16 | CE | – | 81.1 | 92.5 |
| ViT-S/16 | CE | ✓ | 82.6 | 92.7 |
| ViT-S/16 | BCE | – | 83.7 | 92.5 |
| ViT-S/16 | BCE | ✓ | **83.8** | **92.9** |
| ViT-B/16 | CE | – | 84.3 | 92.2 |
| ViT-B/16 | CE | ✓ | 85.1 | 93.5 |
| ViT-B/16 | BCE | – | 85.8 | 93.6 |
| ViT-B/16 | BCE | ✓ | **86.2** | **93.8** |

Masking the same number of randomly chosen patches instead of the most attended ones gives lower accuracy (e.g. ViT-B/16 CIFAR-FS 1-shot: random 84.7 vs. importance-based 86.2), so the gain is not only a regularization effect of CutOut.

**Meta-Dataset** (DINO/IN-1K ViT-B/16, average over 10 domains)

| Training sources | P>M>F | P>M>F + CICO |
|---|---|---|
| 8 in-domain datasets | 84.75 | **85.59** |
| ImageNet only | 79.09 | **80.11** |

## Citation

```bibtex
@inproceedings{jo2025cico,
  author    = {Jo, Seokhee and So, Jungmin},
  title     = {Supervised Contrastive Learning with Importance-based CutOut for Few-shot Image Classification},
  booktitle = {Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing (SAC '25)},
  year      = {2025},
  address   = {Catania, Italy},
  publisher = {ACM},
  doi       = {10.1145/3672608.3707739}
}
```

This code builds on P>M>F; please also cite it:

```bibtex
@inproceedings{hu2022pmf,
  author    = {Hu, Shell Xu and Li, Da and St\"uhmer, Jan and Kim, Minyoung and Hospedales, Timothy M.},
  title     = {Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference},
  booktitle = {CVPR},
  year      = {2022}
}
```

## License

This repository is a modified version of [P>M>F](https://github.com/hushell/pmf_cvpr22) and is released under the same [Apache License 2.0](LICENSE). The original P>M>F code is copyright its authors; the CICO modifications listed above are copyright © 2025 Seokhee Jo.
