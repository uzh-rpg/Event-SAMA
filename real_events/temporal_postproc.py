#!/usr/bin/env python3
"""Temporal-consistency post-processing of per-keyframe segmentation masks.

Input : <run_dir>/Annotations/<vid>/<exp>/<frame>.png  (nearest-keyframe expanded,
        as written by the eval script; the 32 keyframe indices are recovered with
        the same rule the dataset uses: linspace(0, n-1, min(32, n)).astype(int)).
Output: <out_root>/<variant>/Annotations/... for each variant, plus stats.json.

Variants (parameters fixed a priori, NOT tuned on any benchmark):
  v0    identity (re-expansion only; must reproduce the baseline exactly)
  p2    drop components that do not persist across >= 2 consecutive keyframes
  p3    ... >= 3 consecutive keyframes
  p5    ... >= 5 consecutive keyframes
  p3g   p3 + fill single-keyframe gaps (component present at k-1 and k+1, absent at k;
        filled with the k-1 component translated by half the centroid displacement)
  med3  per-pixel majority over keyframes k-1, k, k+1

Component linking is done on a 4x max-pooled grid: components in consecutive keyframes
are linked if the (dilated) one covers >= TAU of the smaller one.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

S = 4          # low-res stride for linking
R = 3          # dilation (low-res px) used for linking
TAU = 0.2      # overlap fraction needed to link two components
VARIANTS = ['v0', 'p2', 'p3', 'p5', 'p3g', 'med3']
FOOT = ndi.generate_binary_structure(2, 2)


def kf_indices(n):
    k = min(32, n)
    if n <= 1:
        return [0] * k
    return [int(i) for i in np.linspace(0, n - 1, k).astype(int)]


def lowres(m):
    H, W = m.shape
    h, w = -(-H // S), -(-W // S)
    pad = np.zeros((h * S, w * S), bool)
    pad[:H, :W] = m
    return pad.reshape(h, S, w, S).max((1, 3))


def up(m_low, shape):
    full = np.repeat(np.repeat(m_low, S, 0), S, 1)
    return full[:shape[0], :shape[1]]


def label_all(Ml):
    labs, ns = [], []
    for m in Ml:
        lab, n = ndi.label(m, structure=FOOT)
        labs.append(lab)
        ns.append(n)
    return labs, ns


def link_graph(labs, ns):
    """Union-find over (k, i). Returns parent dict and list of edges per k."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edges = []
    for k in range(len(labs) - 1):
        a, b = labs[k], labs[k + 1]
        e = []
        if ns[k] and ns[k + 1]:
            area_a = np.bincount(a.ravel(), minlength=ns[k] + 1)
            area_b = np.bincount(b.ravel(), minlength=ns[k + 1] + 1)
            for src, dst, area_s, area_d, ns_s, ns_d, fwd in (
                    (a, b, area_a, area_b, ns[k], ns[k + 1], True),
                    (b, a, area_b, area_a, ns[k + 1], ns[k], False)):
                dil = ndi.grey_dilation(src, footprint=np.ones((2 * R + 1, 2 * R + 1), bool))
                m = (dil > 0) & (dst > 0)
                if not m.any():
                    continue
                pair = dil[m].astype(np.int64) * (ns_d + 1) + dst[m]
                cnt = np.bincount(pair, minlength=(ns_s + 1) * (ns_d + 1)).reshape(ns_s + 1, ns_d + 1)
                for i in range(1, ns_s + 1):
                    for j in range(1, ns_d + 1):
                        c = cnt[i, j]
                        if c and c / max(1, min(area_s[i], area_d[j])) >= TAU:
                            e.append((i, j) if fwd else (j, i))
        e = sorted(set(e))
        for i, j in e:
            union((k, i), (k + 1, j))
        edges.append(e)
    for k, n in enumerate(ns):
        for i in range(1, n + 1):
            find((k, i))
    return find, edges


def persistence(Ml, L):
    K = len(Ml)
    if K < L:
        return [m.copy() for m in Ml], None, None
    labs, ns = label_all(Ml)
    find, edges = link_graph(labs, ns)
    span = {}
    for k, n in enumerate(ns):
        for i in range(1, n + 1):
            span.setdefault(find((k, i)), set()).add(k)
    keep = []
    for k, n in enumerate(ns):
        kept = np.zeros(Ml[k].shape, bool)
        for i in range(1, n + 1):
            if len(span[find((k, i))]) >= L:
                kept |= labs[k] == i
        keep.append(kept)
    return keep, (labs, ns, find), edges


def gapfill(Ml, labs_info, edges):
    """Fill single-keyframe gaps in the (already persistence-filtered) low-res masks."""
    Ml = [m.copy() for m in Ml]
    K = len(Ml)
    if labs_info is None or K < 3:
        return Ml
    labs, ns = label_all(Ml)               # relabel after filtering
    for k in range(1, K - 1):
        la, lb, lm = labs[k - 1], labs[k + 1], Ml[k]
        for i in range(1, ns[k - 1] + 1):
            a = la == i
            if not a.any():
                continue
            da = ndi.binary_dilation(a, structure=FOOT, iterations=2 * R)
            if (da & lm).any():            # something already continues it at k
                continue
            for j in range(1, ns[k + 1] + 1):
                b = lb == j
                if not b.any() or not (da & b).any():
                    continue
                ca = np.array(ndi.center_of_mass(a))
                cb = np.array(ndi.center_of_mass(b))
                moved = ndi.shift(a.astype(np.uint8), (cb - ca) / 2.0, order=0, mode='constant') > 0
                Ml[k] |= moved
                break
    return Ml


def med3(M):
    K = len(M)
    out = []
    for k in range(K):
        if k == 0 or k == K - 1:
            out.append(M[k].copy())
        else:
            out.append((M[k - 1].astype(np.uint8) + M[k] + M[k + 1]) >= 2)
    return out


def apply_keep(M, keep_low):
    out = []
    for m, kl in zip(M, keep_low):
        kl = ndi.binary_dilation(kl, structure=FOOT, iterations=1)
        out.append(m & up(kl, m.shape))
    return out


def make_variants(M):
    Ml = [lowres(m) for m in M]
    res = {'v0': [m.copy() for m in M]}
    for name, L in (('p2', 2), ('p3', 3), ('p5', 5)):
        keep, info, edges = persistence(Ml, L)
        res[name] = apply_keep(M, keep)
        if name == 'p3':
            p3_keep, p3_info, p3_edges = keep, info, edges
    filled_low = gapfill(p3_keep, p3_info, p3_edges)
    p3g = []
    for m0, base, fl in zip(M, res['p3'], filled_low):
        add = up(fl, m0.shape) & ~up(ndi.binary_dilation(lowres(base), structure=FOOT), m0.shape)
        p3g.append(base | add)
    res['p3g'] = p3g
    res['med3'] = med3(M)
    return res


def kf_stats(M):
    nonempty = [bool(m.any()) for m in M]
    area = float(np.mean([m.mean() for m in M]))
    ncomp = [ndi.label(lowres(m), structure=FOOT)[1] for m in M if m.any()]
    ious = []
    for a, b in zip(M[:-1], M[1:]):
        if a.any() and b.any():
            ious.append(float((a & b).sum() / max(1, (a | b).sum())))
    return dict(frac_nonempty_kf=float(np.mean(nonempty)), mean_area=area,
                mean_components=float(np.mean(ncomp)) if ncomp else 0.0,
                mean_consec_iou=float(np.mean(ious)) if ious else None)


def process(args):
    run_dir, out_root, vid, exp, variants = args
    d = f'{run_dir}/Annotations/{vid}/{exp}'
    names = sorted(f for f in os.listdir(d) if f.endswith('.png'))
    n = len(names)
    di = kf_indices(n)
    M = [np.array(Image.open(f'{d}/{names[i]}').convert('L')) > 127 for i in di]
    res = make_variants(M)
    st = {}
    for v in variants:
        Mv = res[v]
        st[v] = kf_stats(Mv)
        od = f'{out_root}/{v}/Annotations/{vid}/{exp}'
        os.makedirs(od, exist_ok=True)
        dia = np.asarray(di)
        for f in range(n):
            j = int(np.argmin(np.abs(dia - f)))
            Image.fromarray((Mv[j].astype(np.uint8) * 255)).save(f'{od}/{names[f]}')
    # identity check against what was on disk (nearest-keyframe rule reproduced exactly)
    ok = True
    for f in (0, n // 2, n - 1):
        j = int(np.argmin(np.abs(np.asarray(di) - f)))
        orig = np.array(Image.open(f'{d}/{names[f]}').convert('L')) > 127
        ok &= bool((orig == res['v0'][j]).all())
    return f'{vid}/{exp}', st, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--procs', type=int, default=16)
    a = ap.parse_args()
    variants = a.variants.split(',')
    items = []
    for vid in sorted(os.listdir(f'{a.run_dir}/Annotations')):
        for exp in sorted(os.listdir(f'{a.run_dir}/Annotations/{vid}'), key=lambda x: int(x) if x.isdigit() else x):
            items.append((a.run_dir, a.out_root, vid, exp, variants))
    print(f'{len(items)} (video, expression) items, variants={variants}', flush=True)
    stats, bad = {}, 0
    with Pool(a.procs) as p:
        for n, (key, st, ok) in enumerate(p.imap_unordered(process, items, chunksize=2)):
            stats[key] = st
            bad += (not ok)
            if n % 100 == 0:
                print(f'  {n}/{len(items)}', flush=True)
    json.dump(stats, open(f'{a.out_root}/stats.json', 'w'), indent=1)
    print(f'done. identity check mismatches: {bad}', flush=True)


if __name__ == '__main__':
    main()
