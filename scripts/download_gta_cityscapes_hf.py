# -*- coding: utf-8 -*-
"""Resumable multi-thread downloader for the GTA5 / Cityscapes HuggingFace mirrors.

Works with both huggingface.co and the hf-mirror.com endpoint (useful where the
former is unreachable). Files already fully downloaded are skipped, so the
script can simply be re-run until everything is present.

Usage:
    python scripts/download_gta_cityscapes_hf.py [--endpoint hf-mirror.com] [--out datasets_dl] [--threads 3]
"""
import argparse
import os
import threading
import time

import requests


def build_file_list(endpoint, out_root):
    MIRROR = 'https://{}/datasets'.format(endpoint)
    files = []
    cs = os.path.join(out_root, 'cityscapes')
    gta = os.path.join(out_root, 'gta5')
    for d in (cs, gta):
        os.makedirs(d, exist_ok=True)

    # Chris1/cityscapes: expected sizes in MiB (validation first = smallest, for a quick start)
    cs_meta = {
        'validation': (3, [398.2, 381.5, 369.7]),
        'train': (14, [464.5, 496.7, 479.5, 471.0, 453.4, 487.7, 504.1,
                       481.4, 470.8, 489.7, 488.0, 491.2, 493.0, 488.1]),
        'test': (7, [449.8, 473.9, 479.6, 483.0, 478.0, 475.6, 472.1]),
    }
    for split, (n, sizes) in cs_meta.items():
        for i in range(n):
            name = '%s-%05d-of-%05d.parquet' % (split, i, n)
            files.append(('%s/Chris1/cityscapes/resolve/main/data/%s' % (MIRROR, name),
                          os.path.join(cs, '%s-%05d.parquet' % (split, i)),
                          int(round(sizes[i] * 2 ** 20))))

    for split, n in [('validation', 30), ('train', 56), ('test', 29)]:
        for i in range(n):
            name = '%s-%05d-of-%05d.parquet' % (split, i, n)
            files.append(('%s/Chris1/GTA5/resolve/main/data/%s' % (MIRROR, name),
                          os.path.join(gta, '%s-%05d.parquet' % (split, i)),
                          0))
    return files


def looks_complete(out, expected):
    if not os.path.exists(out):
        return False
    sz = os.path.getsize(out)
    if expected > 0:
        return abs(sz - expected) < 2 * 2 ** 20  # MiB-rounded sizes -> 2 MiB tolerance
    return sz > 10 * 2 ** 20


class Downloader:
    def __init__(self, n_threads):
        self.rate_limit_until = 0.0
        self.lock = threading.Lock()
        self.n_threads = n_threads
        self.local = threading.local()

    def session(self):
        if not hasattr(self.local, 's'):
            s = requests.Session()
            s.trust_env = False  # ignore system proxy
            self.local.s = s
        return self.local.s

    def download_one(self, url, out, expected):
        s = self.session()
        if looks_complete(out, expected):
            return 'SKIP'
        backoff = 15
        for _ in range(200):
            with self.lock:
                wait = self.rate_limit_until - time.time()
            if wait > 0:
                time.sleep(min(wait, 120))
            try:
                pos = os.path.getsize(out) if os.path.exists(out) else 0
                headers = {'Range': 'bytes=%d-' % pos} if pos else {}
                r = s.get(url, headers=headers, stream=True, timeout=(30, 120))
                if r.status_code == 429:
                    retry_after = int(r.headers.get('Retry-After', backoff))
                    with self.lock:
                        self.rate_limit_until = max(self.rate_limit_until,
                                                    time.time() + retry_after + 5)
                    backoff = min(backoff * 2, 600)
                    continue
                if r.status_code == 416 and os.path.exists(out):
                    if looks_complete(out, expected):
                        return 'DONE'
                    os.remove(out)
                    continue
                r.raise_for_status()
                mode = 'ab' if pos and r.status_code == 206 else 'wb'
                with open(out, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                if looks_complete(out, expected):
                    return 'DONE'
                backoff = 15
            except Exception as e:
                print('[warn]', os.path.basename(out), str(e)[:100])
                time.sleep(backoff)
                backoff = min(backoff * 2, 600)
        return 'FAIL'

    def run(self, files):
        queue = [f for f in files if not looks_complete(f[1], f[2])]
        print('total %d, remaining %d' % (len(files), len(queue)))
        done = [0]

        def worker():
            while True:
                with self.lock:
                    if not queue:
                        return
                    url, out, expected = queue.pop(0)
                res = self.download_one(url, out, expected)
                with self.lock:
                    done[0] += 1
                    print('(%d/%d) %s %s' % (done[0], len(files), res,
                                             os.path.basename(out)))
        threads = [threading.Thread(target=worker) for _ in range(self.n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', type=str, default='hf-mirror.com',
                    help="'hf-mirror.com' or 'huggingface.co'")
    ap.add_argument('--out', type=str, default='datasets_dl')
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    Downloader(args.threads).run(build_file_list(args.endpoint, args.out))


if __name__ == '__main__':
    main()
