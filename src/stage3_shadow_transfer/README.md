# Stage 3 — soft-shadow transfer (comparison)

Real shadows are **soft** (finite light size) and **filled** (indirect light still
arrives) — never a pitch-black binary block. This stage compares three ways to produce
a smooth, filled shadow against a **path-traced ground truth**, on one synthetic concave
testbed (ground + back wall + sphere → contact shadow, concave corner, self-occlusion;
distant **directional** light, matching DiLiGenT).

`shadow_compare.py` renders one method per run; each figure is `binary | method | GT`
across two light directions.

```bash
python src/stage3_shadow_transfer/shadow_compare.py <method> [H] [spp]
#   method ∈ { gifill | prt | sg }
```

| Method | Idea | Expected behaviour |
|---|---|---|
| `gifill` | direct + the **form-factor diffuse bounce** (`gi_operator`) | physically correct **fill** (coloured indirect) in the shadow core; edges stay hard (direct term is still binary). |
| `prt` | per-point **SH visibility transfer** (Sloan 2002) | **soft**, low-frequency shadow — unifies light/dark into one smooth field; can over-soften. |
| `sg` | per-point **single spherical-Gaussian** lobe (bent-normal axis + vMF sharpness, energy-matched) | soft but **sharper** than SH; a single lobe can miss multi-directional occlusion. |

Output: `outputs/rt/shadow/shadow_<method>.png`. The goal is to pick (or combine) the
treatment that best matches the GT's *soft edges + coloured fill* before folding it into
the inverse. Background in
[`../../Markdowns/REMINDERS_NOTES.md`](../../Markdowns/REMINDERS_NOTES.md) (PRT / PhySG).
