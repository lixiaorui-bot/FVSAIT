# -*- coding: utf-8 -*-
"""Pixel-level metrics for the Google Maps <-> Aerial Photo task.

Computes, over the full 1098-image validation set:
  - mean L2 distance (per-pixel RGB Euclidean distance, averaged)
  - pixel accuracy pxAcc(delta) for delta = 30 and 50
    (fraction of pixels whose maximum channel difference is below delta)

Usage (from the repository root, after running test.py):
    python eval/eval_maps_metrics.py --run task2_maps2aerial            # maps -> aerial
    python eval/eval_maps_metrics.py --run task2_aerial2maps --gt data/map/target/val

Pairing: generated files are named <id>_<src_suffix>.jpg in
checkpoints/<run>/images; the ground-truth image is <id>_<tgt_suffix>.jpg in
the reference folder. A = aerial photo, B = Google Map.
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

CFG = {
    'task2_aerial2maps': ('data/map/target/val', 'A', 'B'),
    'task2_maps2aerial': ('data/map/source/val', 'B', 'A'),
}


def evaluate(run_name, gt_dir, src_suffix, tgt_suffix, checkpoints_dir='checkpoints'):
    out_dir = os.path.join(checkpoints_dir, run_name, 'images')
    outs = sorted(glob.glob(os.path.join(out_dir, '*_' + src_suffix + '.jpg')))
    if not outs:
        print('no outputs found in', out_dir)
        return
    l2_list, acc30_list, acc50_list = [], [], []
    missing = 0
    for p in outs:
        base = os.path.basename(p)[: -len('_' + src_suffix + '.jpg')]
        gt_path = os.path.join(gt_dir, base + '_' + tgt_suffix + '.jpg')
        if not os.path.exists(gt_path):
            missing += 1
            continue
        gen = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)
        gt = np.asarray(Image.open(gt_path).convert('RGB')
                        .resize((gen.shape[1], gen.shape[0]), Image.BILINEAR),
                        dtype=np.float32)
        diff = np.abs(gen - gt)
        l2_list.append(np.sqrt((diff ** 2).sum(axis=2)).mean())
        maxc = diff.max(axis=2)
        acc30_list.append((maxc < 30).mean())
        acc50_list.append((maxc < 50).mean())
    n = len(l2_list)
    print('== %s ==  (n=%d, missing gt=%d)' % (run_name, n, missing))
    print('  mean L2 (Dist)      : %.2f +- %.2f' % (np.mean(l2_list), np.std(l2_list)))
    print('  pxAcc (delta = 30)  : %.2f %% +- %.2f' % (np.mean(acc30_list) * 100, np.std(acc30_list) * 100))
    print('  pxAcc (delta = 50)  : %.2f %% +- %.2f' % (np.mean(acc50_list) * 100, np.std(acc50_list) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=str, required=True, help='run name under checkpoints/')
    ap.add_argument('--gt', type=str, default=None,
                    help='ground-truth folder (default: resolved from --run name)')
    ap.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    args = ap.parse_args()
    if args.gt is None:
        gt_dir, src_suffix, tgt_suffix = CFG[args.run]
    else:
        gt_dir = args.gt
        src_suffix, tgt_suffix = ('A', 'B') if '_aerial2maps' in args.run else ('B', 'A')
    evaluate(args.run, gt_dir, src_suffix, tgt_suffix, args.checkpoints_dir)


if __name__ == '__main__':
    main()
