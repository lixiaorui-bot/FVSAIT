# -*- coding: utf-8 -*-
"""Prepare the GTA5 -> Cityscapes dataset from HuggingFace parquet shards.

Shards are the full-resolution image/segmentation pairs mirrored at
    https://huggingface.co/datasets/Chris1/GTA5        (train 56 / validation 30 / test 29 shards)
    https://huggingface.co/datasets/Chris1/cityscapes  (train 14 / validation 3 / test 7 shards)
Download via scripts/download_gta_cityscapes_hf.py (supports hf-mirror.com).

Usage (defaults reproduce the paper split sizes):
    python scripts/prepare_gta_cityscapes.py --datasets_dl datasets_dl --out data

Creates:
    data/source/train  <- GTA5 train shards 00000-00022 (~5100 images, ~5000/epoch in the paper)
    data/source/val    <- GTA5 validation shards 00000-00001 (~430 images)
    data/target/train  <- Cityscapes train shards (2975 images)
    data/target/val    <- Cityscapes validation shards (500 images)

The `image` column stores raw PNG bytes; they are written to disk unchanged.
"""
import argparse
import glob
import os

import pyarrow.parquet as pq


def convert(parquet_files, out_dir, prefix, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    for pf_path in parquet_files:
        if limit is not None and idx >= limit:
            break
        try:
            pf = pq.ParquetFile(pf_path)
        except Exception as e:
            print('  [skip]', os.path.basename(pf_path), str(e)[:60])
            continue
        for batch in pf.iter_batches(batch_size=32, columns=['image']):
            for item in batch.column('image').to_pylist():
                if limit is not None and idx >= limit:
                    break
                out = os.path.join(out_dir, '%s_%05d.png' % (prefix, idx))
                if not os.path.exists(out):
                    with open(out, 'wb') as f:
                        f.write(item['bytes'])
                idx += 1
    print(out_dir, '<-', idx, 'images')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets_dl', type=str, default='datasets_dl',
                    help='directory containing cityscapes/ and gta5/ parquet shards')
    ap.add_argument('--out', type=str, default='data')
    ap.add_argument('--gta_train_shards', type=int, default=23,
                    help='number of GTA5 train shards (~222 images each)')
    ap.add_argument('--gta_val_shards', type=int, default=2)
    args = ap.parse_args()

    gta_train = sorted(glob.glob(os.path.join(args.datasets_dl, 'gta5', 'train-*.parquet')))[:args.gta_train_shards]
    gta_val = sorted(glob.glob(os.path.join(args.datasets_dl, 'gta5', 'validation-*.parquet')))[:args.gta_val_shards]
    cs_train = sorted(glob.glob(os.path.join(args.datasets_dl, 'cityscapes', 'train-*.parquet')))
    cs_val = sorted(glob.glob(os.path.join(args.datasets_dl, 'cityscapes', 'validation-*.parquet')))

    convert(gta_train, os.path.join(args.out, 'data', 'source', 'train'), 'gta')
    convert(gta_val, os.path.join(args.out, 'data', 'source', 'val'), 'gta')
    convert(cs_train, os.path.join(args.out, 'data', 'target', 'train'), 'city')
    convert(cs_val, os.path.join(args.out, 'data', 'target', 'val'), 'city')


if __name__ == '__main__':
    main()
