# -*- coding: utf-8 -*-
"""Prepare the Google Maps <-> Aerial Photo dataset from maps.zip (CycleGAN split).

Usage:
    python scripts/prepare_maps_dataset.py --maps_zip datasets_dl/maps.zip --out data

Creates (A = aerial photo, B = Google Map, following the pix2pix convention):
    data/map/source/train  <- maps/trainA   (1096 aerial)
    data/map/source/val    <- maps/testA    (1098 aerial, validation set)
    data/map/target/train  <- maps/trainB   (1096 map)
    data/map/target/val    <- maps/testB    (1098 map)

Download maps.zip from:
    https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/maps.zip
"""
import argparse
import os
import zipfile

MAPPING = {
    'maps/trainA/': 'data/map/source/train',
    'maps/testA/': 'data/map/source/val',
    'maps/trainB/': 'data/map/target/train',
    'maps/testB/': 'data/map/target/val',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps_zip', type=str, default='datasets_dl/maps.zip')
    ap.add_argument('--out', type=str, default='data',
                    help='project root that contains the data/ directory')
    args = ap.parse_args()

    z = zipfile.ZipFile(args.maps_zip)
    count = 0
    for prefix, out_dir in MAPPING.items():
        members = [n for n in z.namelist()
                   if n.startswith(prefix) and n.lower().endswith(('.jpg', '.png', '.jpeg'))]
        out_root = os.path.join(args.out, out_dir)
        os.makedirs(out_root, exist_ok=True)
        for m in members:
            with z.open(m) as src, open(os.path.join(out_root, os.path.basename(m)), 'wb') as dst:
                dst.write(src.read())
            count += 1
        print(prefix, '->', out_root, ':', len(members))
    z.close()
    print('TOTAL:', count)


if __name__ == '__main__':
    main()
