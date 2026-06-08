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
