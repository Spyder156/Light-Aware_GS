# Light-Decomposed Gaussian Splatting — The Idea, Formally

Intuitive first, then exhaustive and mathematical. This document defines the exact generative model, objective, ambiguity, and disambiguation that the code in [src/](src/) implements and validates. Concept source: [light-aware-gaussian-splatting.md](light-aware-gaussian-splatting.md).

---

## 0. The idea in one breath

A photo is **material × light**. Standard Gaussian Splatting stores their *product* (one color per Gaussian), so it can never tell them apart and breaks when light changes. We instead store **material** on the scene (shared across all photos) and **light** as a small separate model (free to change per photo), and *compute* the photo by physically shading one with the other. Optimize both until renders match the photos. What's left is a **de-lit** scene you can relight.

---

## 1. Image formation (the generative model)

### 1.1 Scene representation

The scene is a set of $N$ Gaussians $\{\mathcal{G}_k\}$. Geometry is standard 3DGS and **frozen/given**: each Gaussian has position $\boldsymbol{\mu}_k$, covariance $\Sigma_k$, opacity, and — crucially for us — a **surface normal** $\mathbf{n}_k$ (fed in from depth/normals, §3 of the concept doc). We **replace the color slot** with **material**:

$$
\mathbf{m}_k = \big(\boldsymbol{\rho}_k,\; r_k,\; \dots\big), \qquad \boldsymbol{\rho}_k \in [0,1]^3 \text{ (albedo)},\; r_k \in [0,1] \text{ (roughness)}.
$$

Material is **shared across all frames** — this is the single most important constraint in the method.

### 1.2 The light menu (per-frame, low-dimensional)

For frame $i$ (camera center $\mathbf{c}_i$, rotation $R_i$), illumination is drawn from a constrained menu:

$$
\underbrace{L^{\text{amb}}_i \in \mathbb{R}^3_{\ge 0}}_{\text{ambient, no position}}, \quad
\underbrace{\big(\mathbf{p}_s,\; \boldsymbol{\ell}^{s}_i\big)}_{\text{static point: shared pos }\mathbf{p}_s,\text{ per-frame color }\boldsymbol{\ell}^s_i}, \quad
\underbrace{\big(\mathbf{o},\; \boldsymbol{\ell}^{t}_i\big)}_{\text{co-moving torch: shared offset }\mathbf{o}}.
$$

The co-moving torch sits at world position $\mathbf{p}^{t}_i = \mathbf{c}_i + R_i\,\mathbf{o}$ (offset $\mathbf{o}$ is in the camera frame, shared across frames; only its color/intensity $\boldsymbol{\ell}^t_i$ varies per frame). On/off is encoded by the magnitude of $\boldsymbol{\ell}$.

**Why a menu.** It is a *parametric prior*. Instead of an arbitrary illumination field (infinite DOF, explains anything), the whole scan's lighting is a handful of shared positions/offsets plus a few per-frame color "dials." This is the cap that forces *detail into material, not light* (concept §2.2).

### 1.3 Shading (the physics that produces a pixel)

For a surface point $\mathbf{x}$ with normal $\mathbf{n}$, albedo $\boldsymbol{\rho}$, seen in frame $i$, the outgoing radiance toward the camera is the sum over active lights of **(light color) × (distance falloff) × (geometry term) × (BRDF)**:

$$
\boxed{\;
L_o(\mathbf{x}) \;=\; \underbrace{\boldsymbol{\rho}\odot L^{\text{amb}}_i}_{\text{ambient}}
\;+\; \sum_{j\in\text{lights}} \boldsymbol{\ell}^{\,j}_i \;\odot\; \underbrace{\frac{1}{d_j^2+\varepsilon}}_{\text{inverse-square}} \;\underbrace{\max(0,\,\mathbf{n}\cdot \hat{\mathbf{l}}_j)}_{\text{Lambert}} \;\Big(\,\underbrace{\boldsymbol{\rho}}_{\text{diffuse}} + \underbrace{f^{\text{spec}}_r(\mathbf{n},\hat{\mathbf{l}}_j,\hat{\mathbf{v}})}_{\text{specular (later)}}\Big)
\;}
$$

where $\hat{\mathbf{l}}_j = \frac{\mathbf{p}_j-\mathbf{x}}{\|\mathbf{p}_j-\mathbf{x}\|}$, $d_j=\|\mathbf{p}_j-\mathbf{x}\|$, $\hat{\mathbf{v}}$ is the view direction, and $\odot$ is per-channel (RGB) product. **Phase 1 uses diffuse only** ($f^{\text{spec}}=0$); specular (a GGX/Cook-Torrance lobe driven by roughness $r$) is added later as a *light probe* cue (concept §6.3).

The final pixel is this radiance, alpha-composited over the Gaussians along the ray (deferred shading on G-buffers $\{\boldsymbol{\rho},\mathbf{n},\mathbf{x},r\}$), then tone-mapped/exposure-scaled:

$$
\hat{I}_i(\mathbf{u}) = \Gamma\!\Big(s_i \cdot \textstyle\sum_k w_k\, L_o(\mathbf{x}_k)\Big), \qquad w_k = \alpha_k \prod_{j<k}(1-\alpha_j),
$$

with per-frame exposure $s_i$ (fixed in Phase 1) and tone-map/clamp $\Gamma$. **This whole map is differentiable** in material and light parameters — that is what lets us invert it.

---

## 2. The inverse problem (what optimization does)

Given photos $\{I_i\}$, known poses, known geometry/normals, solve for **shared material** $\mathbf{M}=\{\mathbf{m}_k\}$ and **per-frame lights** $\Theta=\{L^{\text{amb}}_i, \mathbf{p}_s,\boldsymbol{\ell}^s_i, \mathbf{o},\boldsymbol{\ell}^t_i\}$:

$$
\min_{\mathbf{M},\,\Theta}\; \sum_i \underbrace{\big\| \hat{I}_i(\mathbf{M},\Theta) - I_i \big\|_{\text{photo}}}_{\mathcal{L}_{\text{data}}\;=\;\text{L1} + \lambda\,\text{DSSIM}}
\;+\; \underbrace{\textstyle\sum_q \beta_q\,\mathcal{R}_q(\mathbf{M},\Theta)}_{\text{disambiguation priors (§4)}}.
$$

**The separating force (why this works at all).** Material has one copy shared by *all* frames; lights are re-fit *per* frame. So:
- Whatever is **consistent** across differently-lit views can only be absorbed by material.
- Whatever **varies** frame-to-frame can only be absorbed by lights.
- Because lights are **low-DOF** (the menu), they *cannot* absorb spatial detail → detail is forced into material.

This is a structural argument, not a heuristic.

---

## 3. The central ambiguity (formal statement)

The data term is **invariant to a material↔light rescaling**. For diffuse-only shading, any per-channel positive field $\mathbf{g}\in\mathbb{R}^3_{>0}$ gives:

$$
\boldsymbol{\rho} \mapsto \boldsymbol{\rho}\oslash\mathbf{g}, \qquad \boldsymbol{\ell} \mapsto \boldsymbol{\ell}\odot\mathbf{g}
\quad\Longrightarrow\quad L_o \text{ unchanged.}
$$

(White wall + red lamp) and (pink wall + white lamp) are the same $\mathbf{g}=(\text{red tint})$ choice. **Both render every training photo identically** — so $\mathcal{L}_{\text{data}}$ alone cannot pick the right one. A model can be perfectly *self-consistent* yet *wrong* (concept §6). This is exactly the metamer we visualize in **Step 2**.

**Degrees of freedom view.** A single global $\mathbf{g}$ is 3 numbers. The ambiguity is *low-dimensional*, so it takes only a little extra information to pin down:
- **Multi-frame with varying light** partially fixes $\mathbf{g}$: a $\mathbf{g}$ that explains frame $i$'s light must also explain frame $i'$'s, and the menu can't retune freely.
- **A prior** fixes the rest (§4). One known-white anchor texel collapses $\mathbf{g}$ to a point.

> The ambiguity is a **quality** issue, not a **stability** one: worst case the optimizer lands on *some* self-consistent $\mathbf{g}$ and degrades gracefully to baked-in lighting — it never crashes (concept §6.2).

---

## 4. Disambiguation as regularizers (turning *self-consistent* → *correct*)

Each prior $\mathcal{R}_q$ removes part of the $\mathbf{g}$ freedom. These are *our own* design (we deliberately do **not** import other papers' mechanisms):

1. **Chromaticity / neutrality (Retinex).** Prefer pale material + colored light over colored material + neutral light:
   $$\mathcal{R}_{\text{chroma}} = \sum_k \big\|\boldsymbol{\rho}_k - \bar\rho_k \mathbf{1}\big\|^2,\quad \bar\rho_k=\tfrac13\textstyle\sum_c \rho_{k,c}\quad(\text{penalize albedo saturation}).$$
2. **Reference anchor.** A known-white patch $\mathcal{A}$ pins the scale/color exactly: $\mathcal{R}_{\text{anchor}} = \sum_{k\in\mathcal{A}}\|\boldsymbol{\rho}_k - \rho^{\star}\mathbf{1}\|^2$. Collapses $\mathbf{g}$ to a point — foolproof.
3. **Cross-surface light consistency.** One light's color must explain gradients on *every* surface it touches; faking requires every surface to share a coincidental tint. (Enforced implicitly by a *single shared* $\boldsymbol{\ell}^j_i$ illuminating all Gaussians — already in the model — plus optional smoothness on $\boldsymbol{\ell}$.)
4. **Specular highlights & cast shadows** (later phases): a highlight mirrors the light direction; a shadow localizes the light position. These over-constrain $\mathbf{g}$ geometrically.

**Step 3** shows (1)+(2) selecting the correct factorization where $\mathcal{L}_{\text{data}}$ alone cannot.

---

## 5. Relighting (the payoff)

Once $\boldsymbol{\rho}_k$ (de-lit material) is recovered, **discard** the inferred lights and substitute any new set $\Theta'$:

$$
I^{\text{relit}}(\mathbf{u};\Theta') = \Gamma\!\Big(\textstyle\sum_k w_k\, L_o(\mathbf{x}_k;\,\boldsymbol{\rho}_k,\Theta')\Big).
$$

- $\Theta'=\varnothing$ (all lights off) → the scene "in the dark": pure geometry+material.
- $\Theta'=$ arbitrary lights → physically-consistent relighting (with shadows once the ray tracer is in, concept §7).

**The headline metric** (concept §11, §9): render a *held-out* light position with the recovered material and compare to the true photo under that light — **novel-light PSNR**. Vanilla 3DGS, having baked the training light, gets this badly wrong; we should win clearly. **Step 4** measures exactly this in miniature.

---

## 6. What this minimal validation proves (and what it doesn't)

| Validated here (synthetic, analytic geometry, no GS rasterizer) | Deferred to later phases |
|---|---|
| The shading/forward model is correct (Step 1) | Real Gaussian rasterization (gsplat) |
| The ambiguity is real and data-term-invisible (Step 2) | Ray-traced shadows / relighting (3DGRT) |
| Joint material+light recovery converges (Step 3) | Real captures, auto-exposure, normals-in-the-wild |
| A prior selects the *correct* factorization (Step 3) | Specular/glossy materials, scenes |
| Recovered material relights & beats baked-in on novel-light PSNR (Step 4) | Quantitative benchmarks (Tables 1 & 2) |

We test the **appearance-decomposition math** in isolation, because per concept §2.1 the geometry half of GS is unchanged — only the appearance half needs surgery. If the core decomposition fails here, it fails everywhere; if it holds, the rest is engineering scope.
