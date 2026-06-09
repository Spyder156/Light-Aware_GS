# Related Work — notes for Light-Decomposed Gaussian Splatting

Running index of relevant papers. Each entry: what it is → what (if anything) we take → how it relates to our pitch (§ refs point to [light-aware-gaussian-splatting.md](../light-aware-gaussian-splatting.md)).

> **Stance:** these are reference points, not parts bins. We keep our own design. Borrow *arguments and warnings*, not code or named mechanisms — lifting another paper's machinery makes us look derivative.

---

## RT-Splatting — Joint Reflection-Transmission Modeling with Gaussian Splatting
*Shi, Ying et al., Peking University. arXiv:2605.18263, May 2026.* [PDF](RT-Splatting.pdf) · project: sjj118.github.io/RT-Splatting

**What it is.** Reconstructs thin semi-transparent specular surfaces (car windows, plastic film) where reflection and transmission coexist — a setting where vanilla 3DGS hallucinates floaters. Built on **2DGS**. Splits scene *radiance* into a reflection layer and a transmission layer from one set of Gaussians, under **fixed capture lighting**. No light-source inference, no relighting.

**The relationship in one line.** RT-Splatting separates *light-transport paths* (reflected vs transmitted radiance) on a fixed-lighting scene; it never factors out the illumination and cannot relight. **We factor out the illumination itself and remove it, then relight.** Different problem.

**What we take (three things, nothing more):**

1. **The factorization argument — as a citation, not as code.** They split one overloaded per-Gaussian slot (opacity) into two physically-motivated terms because one slot can't hold a contradiction. That is *structurally the same argument* as our §2.1 (one color slot can't hold material + lighting). Cite once in Related Work as precedent that "factorize the overloaded slot" is a move that works and gets published. **Do not** copy their σ/α (occupancy–opacity) math — that's their problem, not ours.

2. **Their ghost-geometry problem is our warning.** Splitting the slot created a *new* degeneracy (unconstrained "ghost" Gaussians in diffuse regions) that they had to kill with an external supervision loss (a SAM2 mask). The lesson is general and we already half-knew it (§6): **a factorization breeds its own degenerate mode, so plan the anchor before you split.** Our anchors are the chromaticity/neutrality prior + white-card reference. No new mechanism needed — just don't forget to wire the anchor in from day one, not as a patch later.

3. **2DGS base + smartphone-capture scale are validated.** They build on 2DGS and capture real scenes at ~220–240 phone views/scene. Confirms our base choice and own-dataset plan (§8). No action — just confidence.

**What we deliberately do NOT take, and why:**
- *Occupancy–opacity factorization, specular shading network, gradient gating, reflection/transmission split* — all solve their fixed-lighting / semi-transparent problem. None addresses illumination inference or relighting. Importing any of them adds nothing to our contribution and makes us look derivative. Ignore.

**Net:** a one-citation methodological ally that sits on the §9 "Table 1 turf" (consistent capture lighting). It does not encroach on our wedge (material/light decomposition + variable in-the-wild lighting + relighting). Cite it; keep our design as-is.

---

## Path-Traced Inverse Rendering with Global Illumination in 3D Gaussian Fields
*Zhu, Zhang, Zhu, Li, Hu, Gai, Zhu, Huang, Li — USTC + Peking. arXiv:2606.09606, Jun 2026 (concurrent).* [PDF](MrNeRF.pdf) (filed as MrNeRF.pdf)

**What it is.** Splatting-FREE path-traced inverse rendering on 3DGS: forward render + backward
gradients both in one ray-tracing pipeline (no screen-space G-buffers). Recovers albedo/roughness/
metallic + a learnable **Spherical-Gaussian environment light** under **multi-bounce GI** (full
rendering equation, Path Replay Backprop, path-space equivalent surface interaction for overlapping
Gaussians, MIS). New SOTA vs IRGS/SVG-IR/R3DG/GS-IR on TensoIR, Synthetic4Relight, RT4Relight.

**The relationship in one line.** They do **static-environment** path-traced inverse rendering with
GI; we infer & remove **unknown, frame-varying** capture lighting. They assume the one thing we refuse.

**Where it overlaps (acknowledge precisely — do NOT over-concede):**
1. **"GI aids decomposition" is theirs now — but our Step-5 claim is FINER and they did not make it.**
   Their Fig. 10 (Cornell box) shows GI helps disentangle albedo from **color bleeding** (one
   surface's color contaminating a neighbour's albedo). Our Step-5 claim is sharper: multi-bounce
   breaks the **global albedo↔light-color gauge (the metamer)**, demonstrated by **no-prior
   init-collapse** + the **bounce-depth saturation fingerprint**. They still rescale and rely on a
   diffusion prior — i.e. they do NOT claim well-posedness. So: demote Step-5 to a SUPPORTING result
   we cite alongside; do not bury it as scooped. "Multi-bounce makes the light-color decomposition
   well-posed (init-collapse without a prior)" remains a distinct point.
2. **It's a strong build of our planned Phase-4 infra** (ray-traced IR on Gaussians + GI + env-light +
   relighting). The decomposition/GI/ray-tracing ENGINE is now crowded SOTA (this + RT-Splatting).
   Don't compete on the machinery — build on it / cite it.

**Repurpose the GI thread (don't drop it).** It stops being a headline and becomes the **bridge from
their backbone to our wedge**: GI disambiguation only bites in **bounce-rich capture** — exactly the
in-the-wild handheld regime they don't address. That's the justification for the light menu.

**Where it does NOT touch us (the moat):** static single environment light only. Nothing on
variable/unknown lighting DURING capture, the constrained light menu (ambient/static/**co-moving
torch**), per-frame light inference, or in-the-wild handheld capture + dataset.

**Net / strategic:** moat narrowed to the variable-capture-lighting front-end — now the WHOLE
contribution. Cite this (+RT-Splatting) as the path-traced-IR-GS backbone. The single-fixed-light
ceiling we hit in phase1_decompose (~24 dB relight) is the absence of variable lighting, not a tuning
bug — our own experiment and the literature point the same way.
