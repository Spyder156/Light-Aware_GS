"""DiLiGenT-MV full table -- all 5 objects x several views: diffuse vs +GGX, train/novel PSNR.
The paper's first real-data result block. Reuses the verified pipeline (median-ratio albedo,
GGX residual fit, [1,-1,-1] frame, per-light intensity normalization, held-out lights 3..93 step 6).
Saves table to outputs/diligent/table.txt + a summary bar figure.
"""
import os, sys, subprocess, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz

SCENES = ["bearPNG", "buddhaPNG", "cowPNG", "pot2PNG", "readingPNG"]
VIEWS = [1, 8, 15]
OUT = os.path.join(viz.OUT, "diligent")


def main():
    rows = []
    for s in SCENES:
        for v in VIEWS:
            r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "dmv_specular.py"),
                                s, str(v)], capture_output=True, text=True, timeout=1800)
            txt = r.stdout
            m_tr = re.search(r"train: diffuse ([\d.]+) dB -> \+GGX ([\d.]+)", txt)
            m_nv = re.search(r"novel: diffuse ([\d.]+) dB -> \+GGX ([\d.]+)", txt)
            m_mat = re.search(r"mean k_s ([\d.]+) \| mean roughness ([\d.]+)", txt)
            if not (m_tr and m_nv):
                print(f"  {s} v{v:02d}: FAILED\n{txt[-300:]}\n{r.stderr[-200:] if r.stderr else ''}")
                continue
            row = (s, v, float(m_tr.group(1)), float(m_tr.group(2)),
                   float(m_nv.group(1)), float(m_nv.group(2)),
                   float(m_mat.group(1)), float(m_mat.group(2)))
            rows.append(row)
            print(f"  {s:10s} v{v:02d}  train {row[2]:5.1f}->{row[3]:5.1f}  novel {row[4]:5.1f}->{row[5]:5.1f}  "
                  f"ks {row[6]:.2f} rough {row[7]:.2f}")

    # table file
    lines = [f"{'scene':12s} {'view':>4s} {'train diff':>10s} {'train +GGX':>10s} {'novel diff':>10s} "
             f"{'novel +GGX':>10s} {'k_s':>5s} {'rough':>5s}"]
    for s, v, td, tg, nd, ng, ks, ro in rows:
        lines.append(f"{s:12s} {v:>4d} {td:>10.1f} {tg:>10.1f} {nd:>10.1f} {ng:>10.1f} {ks:>5.2f} {ro:>5.2f}")
    arr = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
    lines.append(f"{'MEAN':12s} {'':>4s} {arr[:,0].mean():>10.1f} {arr[:,1].mean():>10.1f} "
                 f"{arr[:,2].mean():>10.1f} {arr[:,3].mean():>10.1f}")
    open(os.path.join(OUT, "table.txt"), "w").write("\n".join(lines))
    print("\n".join(lines[-2:]))

    # summary figure
    labels = [f"{r[0][:-3]} v{r[1]}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - 0.2, arr[:, 2], width=0.4, label="novel: diffuse")
    ax.bar(x + 0.2, arr[:, 3], width=0.4, label="novel: diffuse+GGX")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("held-out novel-light PSNR (dB)")
    ax.set_title("DiLiGenT-MV: real-data relighting of held-out lights (given normals + calibrated lights)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "table_summary.png"), dpi=110); plt.close(fig)
    print(f"-> outputs/diligent/table.txt , table_summary.png")


if __name__ == "__main__":
    main()
