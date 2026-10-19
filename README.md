# FVSAIT: Unpaired Image Translation with FHRR-based Hypervector Encoding and Lookahead Optimization

Official code for the paper
*"FVSAIT: An Improved Image Translation Model using FHRR-based Encoding and Lookahead Optimization"*.

FVSAIT is an unpaired image-to-image translation framework built on a
vector-symbolic-architecture (VSA) hyper-vector space, with two key components:

1. **FHRR-based hypervector encoding** (`models/lsh.py`, `ENCODER: fhrr`).
   Multi-scale VGG19 features are patch-normalized and projected by a random
   (unit-normalized) frequency matrix into the phase domain,
   `theta = pi * conv(x, R)`, i.e. the complex hypervector `z = exp(i*theta)`.
   Binding is phase addition (`vsa_bind`); unbinding (`vsa_unbind`, phase
   subtraction) removes the predicted style from the generated hypervector,
   mapping it back to the source domain for the consistency loss. Similarity
   is the normalized complex inner product
   `mean(cos(theta_1 - theta_2))` (`vsa_similarity`), and the
   per-scale bundling weight is `w_s = 1/S`.
2. **Lookahead optimization** (`utils/lookahead.py`, `LOOKAHEAD: 1`).
   Both the generator and discriminator Adam optimizers are wrapped with the
   Lookahead algorithm (Zhang et al., NeurIPS 2019) with `k = 5`, `alpha = 0.5`;
   the LR scheduler is attached to the inner optimizer.

The generator additionally uses channel attention and self-attention modules
(`models/generators.py`). The original MAP-B / LSH encoder (real-valued random
projection, as in VSAIT) is kept for ablation (`ENCODER: map`).

## Repository layout

```
train.py / test.py / solver.py     entry points and the LightningModule solver
models/                            generator, discriminator, VGG feature extractor, FHRR/LSH encoders
losses/                            VSA + adversarial (hinge) losses
data/                              datasets and joint (source, target) loaders
options/, utils/                   argument parsing, fvcore config, logger, Lookahead
configs/                           GTA -> Cityscapes (model + data + augmentation configs)
configsgoogle/                     Maps <-> Aerial (model + data + augmentation configs)
scripts/                           dataset download / preparation
eval/                              pixel-level evaluation (L2, pxAcc)
```

## Setup

```bash
conda create -n fvsait python=3.8 -y
conda activate fvsait
pip install -r requirements.txt
```

The VGG19 feature extractor (`torchvision`, ImageNet weights) is downloaded
automatically on first use (~548 MB).

## Data preparation

**Google Maps <-> Aerial Photo** (1096 train / 1098 val pairs per direction):

```bash
# download maps.zip (Berkeley server, ~1.4 GB)
python scripts/prepare_maps_dataset.py --maps_zip datasets_dl/maps.zip --out data
```

**GTA -> Cityscapes** (~5100 GTA train / 426 GTA val / 2975 Cityscapes train / 500 Cityscapes val):

```bash
python scripts/download_gta_cityscapes_hf.py --endpoint hf-mirror.com --out datasets_dl
python scripts/prepare_gta_cityscapes.py --datasets_dl datasets_dl --out data
```

## Training

All runs use fixed seed 42 (`pl.seed_everything(42, workers=True)` in `train.py`).

```bash
# Maps -> Aerial Photo (full FVSAIT)
python train.py --name=fvsait_maps2aerial_s42 \
  --model_config=configsgoogle/vsait.yaml \
  --data_config=configsgoogle/google_reverse.yaml \
  --aug_config=configsgoogle/google_aug.yaml \
  --batch_size=4 --max_steps=225000 \
  --val_check_interval=5000 --check_val_every_n_epoch=50 --num_gpus=1

# GTA -> Cityscapes (full FVSAIT)
python train.py --name=fvsait_gta2city_s42 \
  --model_config=configs/vsait.yaml \
  --data_config=configs/gta2cityscapes.yaml \
  --aug_config=configs/gta2cityscapes_aug.yaml \
  --batch_size=1 --max_steps=100000 \
  --val_check_interval=5000 --check_val_every_n_epoch=20 --num_gpus=1
```

### Ablation configs

| config (in `configsgoogle/` or `configs/`) | FHRR encoder | Lookahead | purpose |
|---|---|---|---|
| `vsait.yaml` (default) | yes | yes | full FVSAIT |
| `vsait_mapb.yaml`      | no  | no  | MAP-B baseline (as VSAIT) |
| `vsait_fhrr.yaml`      | yes | no  | FHRR only |
| `vsait_la.yaml`        | no  | yes | Lookahead only |

## Testing and evaluation

```bash
# generate translated images (256x512) for the full validation set
python test.py --name=task2_maps2aerial \
  --checkpoint=checkpoints/<run>/latest.ckpt \
  --model_config=configsgoogle/vsait.yaml \
  --data_config=configsgoogle/google_reverse.yaml \
  --aug_config=configsgoogle/google_aug.yaml --batch_size=1

# pixel metrics (mean L2 distance, pxAcc delta=30/50) over the 1098 paired images
python eval/eval_maps_metrics.py --run task2_maps2aerial

# FID / KID against the real target-domain images
fidelity --gpu 0 --fid --kid --input1 checkpoints/task2_maps2aerial/images \
         --input2 data/map/source/val
```

For the GTA -> Cityscapes task (unpaired), report distributional metrics only
(FID / KID with `--input2 data/target/val`).

## Reproducibility statement

All reported experiments were conducted on a single NVIDIA TITAN Xp (12 GB)
under PyTorch 1.13.1 (CUDA 11.7) and pytorch-lightning 1.9.5. The random seed
for the main experiments is 42; the FHRR projection matrix `R` is a registered
buffer and is stored inside every checkpoint.

## License and acknowledgment

This code is derived from the VSAIT implementation
(https://github.com/facebookresearch/vsait) and is released under the same
CC BY-NC 4.0 license (see `LICENSE`).
