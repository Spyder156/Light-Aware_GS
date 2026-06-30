"""Visualization helpers -- save labeled image panels to outputs/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)


def to_np(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float().numpy()
    return np.clip(img, 0, 1)


def normal_to_rgb(n):
    n = to_np(n)
    return np.clip(0.5 * (n + 1.0), 0, 1)


def _path(fname, subdir=None):
    d = os.path.join(OUT, subdir) if subdir else OUT
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def panel(images, titles, fname, cols=None, cmaps=None, suptitle=None,
          figsize_scale=3.0, subdir=None):
    """images: list of HxWx3 (or HxW). Saves a single labeled figure to OUT/[subdir]/fname."""
    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            im = to_np(images[i])
            cm = (cmaps[i] if cmaps else None)
            if im.ndim == 2:
                ax.imshow(im, cmap=cm or "viridis")
            else:
                ax.imshow(im)
            ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    path = _path(fname, subdir)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def curve(xs, ys_dict, xlabel, ylabel, fname, title=None, logy=False, subdir=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, ys in ys_dict.items():
        ax.plot(xs, ys, label=label, marker="o", ms=3)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    path = _path(fname, subdir)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
