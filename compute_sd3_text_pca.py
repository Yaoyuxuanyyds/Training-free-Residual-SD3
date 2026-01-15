#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render PCA plots from dumped SD3 text embeddings."""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


def load_layer_features(npz_path: str):
    data = np.load(npz_path)
    if "layers" in data:
        layers = [int(x) for x in data["layers"].tolist()]
    else:
        layers = sorted(
            int(k.replace("layer_", ""))
            for k in data.files
            if k.startswith("layer_")
        )
    feats_by_layer = {}
    for layer in layers:
        key = f"layer_{layer}"
        if key in data:
            feats_by_layer[layer] = data[key]
    return layers, feats_by_layer

def normalize_density(density: np.ndarray, gamma: float = 0.5):
    """
    Perceptually enhanced density normalization.
    """
    # log compression (already exp-ed, so do log1p)
    d = np.log1p(density)

    # robust clipping (ignore extreme outliers)
    lo, hi = np.percentile(d, [2, 98])
    d = np.clip(d, lo, hi)

    # normalize
    d = (d - lo) / (hi - lo + 1e-12)

    # optional gamma (boost mid-density contrast)
    d = d ** gamma
    return d


def compute_density(points: np.ndarray, bandwidth: float | None):
    if bandwidth is None:
        # Silverman-like heuristic
        std = np.std(points, axis=0).mean()
        bandwidth = 0.15 * std

    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(points)
    log_density = kde.score_samples(points)
    return np.exp(log_density)



def plot_layer(points, output_path, bandwidth, cmap, point_size):
    density = compute_density(points, bandwidth=bandwidth)
    if density.size == 0:
        return

    density = normalize_density(density, gamma=0.8)

    order = np.argsort(density)
    points = points[order]
    density = density[order]

    plt.figure(figsize=(3, 3), dpi=200)
    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=density,
        s=point_size,
        cmap=cmap,
        alpha=0.9,
        linewidths=0,
    )
    plt.axis("off")
    plt.margins(0, 0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()



def run(args: argparse.Namespace):
    layers, feats_by_layer = load_layer_features(args.npz_path)
    if not feats_by_layer:
        raise ValueError("No layer features found in the PCA dump.")

    all_feats = [feats for feats in feats_by_layer.values() if feats.shape[0] > 2]
    if not all_feats:
        raise ValueError("Not enough tokens to run PCA.")

    pca_fit = np.concatenate(all_feats, axis=0)
    if pca_fit.shape[0] < 3:
        raise ValueError("Not enough tokens to run PCA.")

    pca = PCA(n_components=2, random_state=42).fit(pca_fit)

    os.makedirs(args.output_dir, exist_ok=True)
    for layer in layers:
        feats = feats_by_layer.get(layer)
        if feats is None or feats.shape[0] <= 2:
            continue
        pca_2d = pca.transform(feats)
        output_path = os.path.join(args.output_dir, f"{args.output_prefix}_layer_{layer:02d}.png")
        plot_layer(
            pca_2d,
            output_path=output_path,
            bandwidth=args.density_bandwidth,
            cmap=args.cmap,
            point_size=args.point_size,
        )
        print(f"[SAVE] {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render per-layer PCA point clouds from SD3 text embeddings.")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to PCA dump produced by compute_sd3_text_exp.py")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save PCA images")
    parser.add_argument("--output-prefix", type=str, default="text_pca", help="Prefix for PCA images")
    parser.add_argument(
        "--density-bandwidth",
        type=float,
        default=None,
        help="KDE bandwidth (None = adaptive)"
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for density (blue->yellow)")
    parser.add_argument("--point-size", type=float, default=4.0, help="Scatter point size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.npz_path) or "."
    run(args)
