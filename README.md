# FVSAIT

FVSAIT: unpaired image translation with FHRR-based hypervector encoding and
Lookahead optimization.

## Requirements

- Python 3.8, PyTorch 1.10, pytorch-lightning 1.5 (see `requirements.txt`)

## Training

```bash
# GTA -> Cityscapes
python train.py \
  --name fvsait_task1 \
  --data_config ./configs/task1_gta2city_data.yaml \
  --model_config ./configs/fvsait.yaml \
  --aug_config ./configs/gta2cityscapes_aug.yaml \
  --output_dir ./checkpoints --batch_size 1 --seed 42 --max_epochs 20 \
  --val_check_interval 2000

# Aerial Photo -> Google Maps
python train.py \
  --name fvsait_task2_p2m \
  --data_config ./configs/task2_photo2map_data.yaml \
  --model_config ./configs/fvsait.yaml \
  --aug_config ./configs/square256_aug.yaml \
  --output_dir ./checkpoints --batch_size 4 --seed 42 --max_epochs 400 \
  --val_check_interval 2000
```

## Testing

```bash
python test.py \
  --name fvsait_test \
  --checkpoint ./checkpoints/fvsait_task1/version_0/checkpoints/last.ckpt \
  --data_config ./configs/task1_gta2city_data.yaml \
  --model_config ./configs/fvsait.yaml \
  --aug_config ./configs/gta2cityscapes_aug.yaml \
  --batch_size 1 --num_gpus 1 --output_dir ./results
```

## Ablation configs

| config | encoding | optimizer |
|---|---|---|
| `configs/fvsait_mapb_adam.yaml` | MAP-B | Adam |
| `configs/fvsait_mapb_lookahead.yaml` | MAP-B | Adam + Lookahead |
| `configs/fvsait_fhrr_adam.yaml` | FHRR | Adam |
| `configs/fvsait.yaml` | FHRR | Adam + Lookahead |
