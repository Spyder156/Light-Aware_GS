# Light-Decomposed Gaussian Splatting

### Relightable 3D reconstruction from in-the-wild captures under unknown, variable lighting

> **One-line idea:** Standard Gaussian Splatting bakes lighting into the scene and breaks when the lighting changes during capture (e.g. someone scanning with a torch in hand). Instead of fighting that, we give the optimizer the freedom to *explain each image's lighting separately* using a small, constrained menu of light sources — while forcing the scene's *material* to stay shared across all frames. The result is a "de-lit" 3D model: pure geometry + material with no lighting baked in, which can then be relit with any lights you want.

---

## 1. The Problem

When you reconstruct a scene from photos with Gaussian Splatting (or any photo-based 3D method), the software quietly assumes **the lighting never changes** across the capture.

In the real world this assumption breaks constantly:

- Someone walks around a room with a torch/flashlight in hand.
- A phone scan where the phone's own light moves with the camera.
- Daylight shifts, a lamp gets switched on, a cloud passes.

When lighting varies frame-to-frame, the reconstructor cannot tell the difference between:

- *"This wall **looks** brighter in this photo"* (because the light moved), and
- *"This wall **is** painted brighter"* (real material).

So it gives up and **bakes the lighting into the scene as permanent color** — producing smudges, wrong colors, floating "floater" Gaussians, and a generally broken reconstruction.

**We want two things that standard GS cannot give:**

1. **Robustness:** reconstruction should not break under variable lighting during capture.
2. **Relighting:** the final model should have lighting *separated* from material, so we can switch all lights off (see the scene's true geometry/material "in the dark") and then add arbitrary new lights — any color, any position.

---

## 2. The Core Idea (and the Ideology Behind It)

### 2.1 Don't fight the lighting — model it explicitly

Vanilla Gaussian Splatting learns **lit appearance**. Each Gaussian stores essentially *one color* — the color that point happens to show. Lighting is already fused into that color; GS never separates "what the surface is" from "how it's lit," because it has only one slot to store both.

That single slot is *exactly why it breaks* under variable light: the same point shows different colors in different frames, and one slot can't hold a contradiction.

**The fix is structural, not incidental.** To separate lighting, the representation needs a *separate home* for it:

- Each Gaussian stores **material** (true surface properties: albedo/color, roughness, etc.) — **shared across all frames**.
- A **separate lighting model** describes the lights — allowed to **vary per image**.
- The rendered color is then *computed*: `material ⊗ light`, via ray-traced / physically-based shading.

> The geometry half of GS (learning shape from posed images) stays as-is. It works fine. **Only the appearance half needs surgery** — swap the "color" slot for "material + an explicit, separate light model."

This is a research build: we will build whatever architecture and code is required. The point of the project is precisely to construct this separated representation, not to bolt onto an existing one.

### 2.2 The separation mechanism: shared vs. per-frame

Why does this actually separate light from material instead of just shuffling the ambiguity around? Two forces pulling against each other:

1. **Material is shared.** The same patch of wall appears in many frames, each lit differently, and must reconcile to **one** material. Whatever is *consistent* across frames flows into material; whatever *varies* flows into lighting.
2. **Lighting is deliberately simple.** We do **not** allow arbitrarily complex per-image lighting. We give each image only a small, constrained budget of lights. If lighting could be arbitrarily rich, it could explain *anything* and material would learn nothing. By forcing lighting to be simple, anything *detailed* in the image must be explained by material.

> **Simple lights, detailed surfaces.** That budget cap is the whole trick.

### 2.3 The constrained light menu

Rather than "any possible lighting" (a nightmare) or "one fully-known torch" (too rigid), we give the optimizer a **menu of light types** and let it fill in the parameters per image:

- **Ambient** — a global, position-less glow (per-image intensity/color).
- **Static torch / point light** — a light fixed at some world position (one shared position; per-image on/off + intensity).
- **Torch moving with the camera** — a light rigidly attached to the camera with a fixed (shared, unknown) offset; per-image intensity.

We do **not** assume we know where any light is, or whether it moves with the camera. The optimizer figures out which lights are active and how strong, per image. The menu is a **parametric prior** that collapses the search space to a tiny number of unknowns (a couple of positions/offsets shared across the whole scan, plus a few intensity "dials" per image).

This menu is also a key part of the *disambiguation* story (Section 6): because the model can only reach for *simple, structured* lights, it cannot invent weird lighting to cheat with.

---

## 3. What We Need (Inputs)

- **The scan**: images / video from a handheld capture under variable lighting.
- **Camera poses** per frame (standard output of any scan / SfM / SLAM pipeline).
- **Geometry — depths and normals.** We assume we can obtain clean, accurate depths and normals (e.g. from a depth sensor / LiDAR phone, or estimated and fused), and feed them in. Normals are important: they enable proper physically-based shading and ray-casting (you need surface orientation to compute how light hits a surface).

**Notably absent:** any information about the lights. Light positions, colors, and intensities are *unknowns we solve for*, not inputs we require.

---

## 4. What the Model Does (Conceptually)

1. Build one shared 3D model where every surface (Gaussian) has an **unknown material** and a **known normal/geometry**.
2. Attach to each image its own small set of **unknown lights**, drawn from the constrained menu (ambient / static / co-moving).
3. For each frame: **render** it by ray-casting from that frame's lights, through the shared material and known geometry, to the camera — and **compare** to the real photo.
4. **Optimize everything jointly** across all frames at once: adjust materials (shared) and per-frame light parameters until renders match the real photos.

Because material is shared and lighting is per-frame, the optimization is *forced* to sort "what's actually there" from "how it happened to be lit."

**Output:** a **de-lit 3D model** — pure geometry + material, no lighting baked in. From there:

- Turn all lights off → see the scene's true geometry/material "in the dark."
- Add any lights you want, anywhere, any color → relighting.

Every Gaussian already carries the parameters needed to be relit, so the relit renders are physically consistent.

---

## 5. Why Even a Single Image Is Informative

A common worry is that "any possible lighting" makes the problem hopeless. It's less hopeless than it sounds, because lighting leaves strong cues even in one frame:

- **Shading falloff across a curved surface** directly encodes the **light direction** (classic shape-from-shading). A single curved surface already pins down where light is roughly coming from.
- **Inverse-square falloff** (a surface dims with distance from a point light) constrains the light's **position/distance**.
- **Specular highlights** are a mirror of the light — one highlight on a glossy surface gives its direction almost exactly.
- **Cast shadows** localize light positions and prove surfaces aren't self-emissive.

So a single image gives a lot — especially light *direction*. The one thing a single image *cannot* settle on its own is the **scale** of the material-vs-light split (see Section 6). That's what multiple frames (or strong priors) are for.

---

## 6. The Central Subtlety: "Correct" vs. "Self-Consistent"

This is the hardest and most important issue in the whole project.

### 6.1 The ambiguity

Imagine a **white wall lit by a red lamp**. The wall looks pink in every photo. The model can explain this two ways:

- **Correct:** wall is *white*, lamp is *red*. → Turn lamp off → white wall. Add a blue lamp → blue wall.
- **Wrong but self-consistent:** wall is *pink*, lamp is *white*. → Turn lamp off → still pink. Add a blue lamp → purple wall.

**Both render the training photos identically.** From the photos alone, you cannot tell which factorization the model landed on. A model can be perfectly *self-consistent* (renders every training view correctly) and still be *wrong* (picked the bad factorization), which ruins relighting.

> The training data can't prove correctness. This is the thing reviewers will probe, and it's the quality axis the paper lives or dies on.

### 6.2 Important nuance: this is a *quality* issue, not a *stability* issue

Crucially, the ambiguity does **not break** the reconstruction:

- If lighting were constant across all frames, the model simply converges to constant light parameters; training views render fine; you can still toggle and add lights.
- Worst case, it degrades *gracefully* to what baseline GS does anyway (lighting baked in). It never crashes.

So variable lighting is **not a hard requirement** for the method to *function*. It's one of two routes to a *correct* decomposition; the other route is a strong enough prior. We have priors (the constrained light menu, plus the ones below).

### 6.3 The disambiguation toolkit

We resolve "correct vs. self-consistent" with a combination of the following — no single one is sufficient, the *combination* is what makes faking the decomposition essentially impossible:

- **Cross-surface light consistency.** Force *one* light's color/position to explain the gradients on *every* surface it touches (wall + floor + ceiling) simultaneously. Faking this would require every surface in the room to be coincidentally the same tint — improbable. Real lights are economical explanations; fake ones require conspiracies.
- **Light localization from falloff.** If a single light position + color explains the inverse-square falloff and gradients across multiple surfaces at once, you've found the real light. (Limitation: ambient light has no position, so it stays partly ambiguous — that's what the chromaticity prior is for.)
- **Specular highlights as light probes.** A glossy surface mirrors the light; a shiny object in the scene is a free light probe pinning down direction.
- **Cast shadows.** Once ray tracing is in place, shadows hard-localize light positions and prove surfaces are externally lit, not self-emissive.
- **Chromaticity / neutrality prior on albedo.** When ambiguous, prefer *pale surfaces + colored lights* over *colored surfaces + neutral lights* (the classic Retinex / intrinsic-decomposition assumption). Cheap and surprisingly effective.
- **Reference anchors.** A single known-white patch (a sheet of paper, a calibration card) breaks the scale/color ambiguity instantly. Foolproof and reviewer-friendly.
- **Lighting variation as an optional tool.** If the capturer toggles the torch on/off for even a few frames, lighting changes while material doesn't — the ambiguity collapses directly. Optional capture protocol, not mandatory.

**Strongest combo for our setup:** cross-surface consistency + highlights + shadows + chromaticity prior. With all four, the "pink wall / white lamp" fake can't simultaneously explain a glossy reflection, a cast shadow, *and* every other surface in the room.

---

## 7. Infrastructure: Ray-Traced Gaussian Splatting

To compute `material ⊗ light` properly — with shadows, secondary bounces, and arbitrary light models — we build on **ray-traced Gaussian Splatting (3DGRT-style)**, which traces rays through Gaussian primitives (hardware-accelerated, k-buffer hit marching). Recent inverse-rendering / relighting work on Gaussians (e.g. IRGS integrating the full rendering equation, RaySplats) relies on having such an efficient ray tracer.

**Where ray tracing earns its place:** at *relighting* time, new lights live at arbitrary positions, so shadows and inter-reflections genuinely matter — that's the full rendering equation. During *capture-time decomposition*, the dominant signal is direct illumination from the inferred lights, which is cheaper. The ray tracer is the plumbing for the relighting demo and for shadow-based disambiguation; the *contribution* is the separated material+light representation and the priors that make the decomposition well-posed, not the ray tracer itself.

---

## 8. Datasets

**No public dataset is exactly our scenario** (a handheld scan under unknown, variable lighting). Capturing our own is therefore both necessary and a contribution in its own right (a phone + a torch is enough; flash/no-flash paired frames give cheap decomposition ground truth — the no-flash frame is the ambient, the difference is the pure torch contribution).

### Objects (start here)

- **OpenIllumination** — real objects under many *known* light positions; built for inverse rendering. Primary quantitative benchmark.
- **Stanford ORB (Object Relighting Benchmark)** — real captures with HDR env-map ground truth; standard for relighting papers.
- **ReNe** — real objects under varying lighting.
- **NeRFactor Synthetic4Relight** — Blender objects with full GT (albedo, normals, relighting).

### Scenes (later)

- **LuxRemix synthetic indoor scenes** (~10k procedurally lit scenes, Jan 2026) — large-scale multi-light synthetic.
- **NeRF-OSR** — outdoor scene relighting under varying illumination.
- **Mip-NeRF 360** — geometry / novel-view baseline (no relighting GT).

### How to get ground-truth relighting on real data without a light stage

For each virtual camera in an OLAT (one-light-at-a-time) dataset, pick the light nearest that camera to **simulate a co-located moving torch**, holding out other lights as relighting ground truth. This yields real-data relighting metrics without building a rig. Pair this with our own handheld captures for the in-the-wild demo.

---

## 9. Baselines & Evaluation

The framing is a **two-table story**:

**Table 1 — On their turf (static lighting at capture, standard benchmarks):**
Be *competitive* (don't need to win) against the established relightable-GS methods: **Relightable 3DGS, GaussianShader, GS-IR, IRGS, TensoIR**.

**Table 2 — On our turf (variable lighting at capture — our new setting):**
Here the established methods **collapse**, and we don't. This is the headline. Include **vanilla 3DGS** (shows it breaks) and **GS-W / WildGaussians** (appearance embeddings — survive variable appearance but cannot relight) as ablation baselines.

> The paper's one-line framing: *"We match SOTA where they work, and we work where they don't."*

**Metrics:** novel-view synthesis under held-out light positions (PSNR/SSIM/LPIPS); albedo error vs GT; light position/color error; relighting error vs GT env maps; decomposition cleanness. The **ambient-strength sweep** (pitch dark → dim → bright ambient) on synthetic data is the money plot: it shows exactly when the decomposition holds and when it degrades.

---

## 10. The Hero Demo

The one thing a reviewer remembers:

> **A messy handheld scan → the lights stripped out, revealing the scene's geometry/material sitting in total darkness → the same scene relit however we want (sunset, a lamp in the corner, a colored light anywhere).**

If that "scan → pitch-black geometry → arbitrary relight" video works cleanly, *that is the paper.* Videos sell relighting work; this demo is the centerpiece.

---

## 11. Scope & Roadmap

- **Phase 1 — Objects.** Start here: easier, and standard for relighting work. Validate the core decomposition + relighting on OpenIllumination / ORB.
- **Phase 2 — Scenes.** Extend to full scenes (harder eval, larger captures).

**Suggested first milestone (before writing anything):** synthetic object, locked exposure, torch-dominant (minimal ambient), diffuse material only. Get the separated model to beat vanilla 3DGS on *novel-light-position PSNR*. If that single number moves, the idea is real and everything else (auto-exposure handling, ambient, glossy materials, full relighting) is incremental scope. If it doesn't move, you find out in a week, not a month.

---

## 12. Honest Paper-Tier Assessment

*(Acceptance is ~10% idea, ~90% execution. This assumes we actually achieve what we claim.)*

- **Solid CVPR accept** — very plausible if executed well.
- **Highlight / Spotlight** — reachable but **not automatic**. Needs the demo videos to genuinely wow *and* clean, large quantitative wins on the variable-lighting setting.
- **Oral** — reach territory. Relightable GS is a *crowded* space, so reviewers will scrutinize the delta-over-prior hard. Our wedge ("variable-lighting in-the-wild capture") is real and underserved, which is the strongest argument. Two things would push toward oral:
  1. **Release a benchmark dataset** for variable-lighting capture. Adopted datasets get cited; cited work gets orals.
  2. **Make the hero demo undeniable.** The "scan → dark geometry → arbitrary relight" video, if it lands, is exactly the kind of result that earns an oral.

> Realistic ceiling: solid accept with a real shot at highlight. Oral is achievable but must be *earned* via the dataset contribution + a demo that lands hard.

---

## 13. FAQ — Common Doubts & Answers

**Q: Gaussian Splatting already reconstructs geometry from scratch (just from cameras). Can't it just learn the lighting too, the same way?**
Not directly. GS learns *lit appearance* — a single color slot per Gaussian, with lighting already fused in. It has no separate place for lighting, so there's nothing to "separate." You must change what each Gaussian stores: swap the color slot for *material*, and add a *separate, explicit* light model. The geometry-learning part stays unchanged; only the appearance part needs the structural change.

**Q: But we don't know where the torch is, or even whether it moves with the camera.**
Correct — and we don't assume we do. Each image gets a small budget of *unknown* lights from a constrained menu (ambient / static / co-moving). The optimizer infers their parameters per frame. We solve for the lights; we don't require them as input.

**Q: Doesn't giving the model freedom to "invent lights" let it cheat?**
That's prevented by two opposing constraints: (1) material is *shared* across all frames (consistent stuff → material, varying stuff → lighting), and (2) the light budget is deliberately *small/simple*, so detailed image content must be explained by material, not by inventing rich lighting. Simple lights, detailed surfaces.

**Q: Doesn't this only work if the lighting varies across frames?**
No. With constant lighting, the model just learns constant light parameters; training views render fine; you can still toggle/add lights. Variation is *not required* for the method to function. It's one route to a *correct* decomposition; a strong prior (our light menu + chromaticity prior + anchors) is the other. We have the priors.

**Q: What about the "bright paint vs. bright light" ambiguity — does it need special handling?**
No special treatment. Worst case it degrades gracefully to what baseline GS does anyway (lighting baked in) — it never breaks the reconstruction. It's a *quality* issue (how good the relighting is), not a *stability* issue. We address quality with the disambiguation toolkit (Section 6.3).

**Q: How do you prove the split is *correct*, not merely *self-consistent*?**
This is the crux. A white wall under a red lamp looks pink; the model could decide "white wall + red lamp" (correct) or "pink wall + white lamp" (wrong) — both render the photos identically. The training data alone can't tell them apart. We prove correctness with (a) ground-truth datasets (synthetic + known-light benchmarks) where the true albedo/lights are known, and (b) the disambiguation toolkit that makes the wrong factorization physically inconsistent across surfaces, highlights, and shadows.

**Q: Won't the cast rays + inverse-square falloff just localize the light source and resolve this?**
Partly — yes for *positioned* lights. If one light position + color explains the falloff and gradients across multiple surfaces at once, you've found the real light, and that's hard to fake. But ambient light has no position, so it stays ambiguous; that residual is handled by the chromaticity/neutrality prior and reference anchors. Falloff is a strong cue, not a complete solution on its own.

**Q: Do we feed in normals?**
Yes. We assume clean normals/depths are available and feed them in. Normals are required for physically-based shading and ray-casting — you need surface orientation to compute how light hits each surface.

**Q: Why ray tracing (3DGRT) if a co-located light casts no visible shadows anyway?**
Ray tracing earns its place at *relighting* time: new lights sit at arbitrary positions, so shadows and inter-reflections matter (full rendering equation). It's also what makes shadow-based disambiguation possible. During capture-time decomposition it's less critical. The ray tracer is infrastructure; the contribution is the separated representation + priors.

**Q: What's the difference from existing relightable-GS / photometric-stereo-GS work?**
Existing work (Relightable 3DGS, IRGS, GS-IR, PS-GS, etc.) generally assumes *consistent/known* lighting at capture, or controlled multi-light lab setups. Our wedge is *unknown, variable, in-the-wild* lighting during a single handheld scan — a capture scenario those methods break on. We match them on their settings and work on ours.

---

## 14. Open Decisions (still to lock)

- Final one-sentence pitch for the abstract.
- Exact light-budget policy (how many lights per image; fixed vs. adaptive).
- Whether normals stay frozen or are lightly refined in a later stage.
- Real-capture protocol details (locked vs. auto exposure; flash/no-flash pairing; reference-anchor inclusion).
- Whether to formalize and release the variable-lighting benchmark dataset (strongly recommended for tier).
