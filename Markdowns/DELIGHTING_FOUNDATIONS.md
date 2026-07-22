# De-Lighting in the Wild: Mathematical Foundations

*A toolbox of physical models, identifiability arguments, and math tricks for fully stripping illumination from a general captured scene — lab rig or casual 360° capture — by decomposing light into a small dictionary of source types. Companion to `math.md`; notation carried over.*

---

## 0. What changes outside the lab

`math.md` assumed the ideal experiment: known geometry, calibrated narrowband sensing, and $K$ lamp positions under our control. A wild capture (Mip-NeRF-style: walk around, one fixed unknown illumination) differs in four ways:

1. **$K = 1$.** The illumination never changes. All the leverage we got from *moving the lamp* (configuration differencing, per-point rank-3 systems) is gone and must be replaced by *spatial* structure.
2. **But coverage is better.** A 360° capture photographs the ground, walls, trees — the very surfaces DiLiGenT masks out. The near-field radiosity $B$ is measured almost everywhere, and often the *sky itself* is in frame.
3. **Sensing is worse.** RGB (3 broad channels, metamerism), per-frame auto-exposure/white-balance nuisances, tone curves.
4. **Geometry is recovered, not given** — with its own error budget.

The goal of this document: the mathematical foundation that makes de-lighting solvable in this regime — the light model, the ambiguity ledger, the evidence that pins each unknown, and the estimation architecture.

---

## 1. The central principle: split transport at the reconstruction boundary

Incident light at a surface point $x$ is an integral over the hemisphere. Partition directions by *what the reconstruction knows about them*:

$$
E(x) \;=\; \underbrace{\int_{\omega \,\to\, \text{hits reconstructed surface}} \frac{B_{\text{meas}}(h(x,\omega))}{\pi}\,\cos\theta \, d\omega}_{\text{near field: } H(x),\ \textbf{measured}}
\;+\; \underbrace{\int_{\omega \,\to\, \text{visible sky pixels}} L_{\text{sky}}^{\text{meas}}(\omega)\,\cos\theta \, d\omega}_{\textbf{measured far field}}
\;+\; \underbrace{\int_{\omega \,\to\, \text{unobserved}} L_{\text{far}}(\omega;\theta)\,\cos\theta \, d\omega}_{\textbf{parametric far field (the dictionary)}}
$$

Three consequences, each doing real work:

- **The near field needs no model.** As in `math.md` §5.1, bounce light from any reconstructed surface is a gather over *measured* radiosity. In a 360° capture this is the *largest* term (ground bounce, wall bounce) — the wild scene is, ironically, better suited to the exact trick than DiLiGenT is.
- **Emitters inside the scene are free.** The gather does not care *why* a surface is bright. A visible lamp's glow enters $H$ automatically through its measured $B$. (Its sharp direct contribution can additionally get a point-source dictionary element for accuracy; see §2.4, and §7.6 for detecting emitters.)
- **The dictionary only has to cover what was never seen.** Sun behind the camera, sky occluded by buildings, the room outside the captured volume. This is what keeps the parametric part *low-dimensional* — the key to identifiability in §5.

---

## 2. The light dictionary

Each element: physical model → irradiance footprint $g(x)$ (a precomputable geometric function, given geometry) → the image evidence that identifies it → its failure/degeneracy mode. The total direct-plus-far model is

$$
E(x) \;=\; H(x) + \sum_s S_s \, g_s(x;\, \xi_s),
$$

with $S_s$ the (per-channel) source intensity/color — **linear** unknowns — and $\xi_s$ geometric parameters (directions, positions) — **nonlinear** unknowns.

### 2.1 Uniform ambient

$$
L_{\text{far}}(\omega) = L_a \quad\Longrightarrow\quad g_a(x) = \pi A(x), \qquad A(x) = \tfrac{1}{\pi}\!\int V_{\text{env}}(x,\omega)\cos\theta\, d\omega .
$$

*Evidence:* darkening in creases and cavities exactly proportional to $1{-}A(x)$ — a field computable from geometry alone. *Degeneracy:* on open flat scenes $A \approx \text{const}$, so ambient trades off against global albedo scale; needs the anchors of §5.

### 2.2 Directional-ambient (graded sky) — spherical harmonics to order 2

The general far field expanded in real spherical harmonics: $L_{\text{far}}(\omega) = \sum_{l,m} c_{lm} Y_{lm}(\omega)$. For **unshadowed** Lambertian response there is a clean theorem: irradiance is the spherical convolution of $L_{\text{far}}$ with the clamped-cosine kernel, whose spectrum decays fast —

$$
E(\hat n) = \sum_{l,m} A_l \, c_{lm} \, Y_{lm}(\hat n), \qquad A_0 = \pi,\; A_1 = \tfrac{2\pi}{3},\; A_2 = \tfrac{\pi}{4},\; A_l \approx 0 \ (l \ge 3 \text{ odd}),\ \sim l^{-2} \text{ decay},
$$

so **9 coefficients capture ≳99% of diffuse shading from any far field**. Your "directional ambient" is exactly the $l{=}1$ band (a cosine gradient: bright sky above, dark ground below); $l{=}2$ adds the quadratic term. Two crucial caveats that are *features*:

- With **occlusion**, $E(x) = T(x)^\top c$ where the transfer vector $T(x) = \big[\int V_{\text{env}}(x,\omega) Y_{lm}(\omega)\cos\theta\, d\omega\big]_{lm}$ is precomputable per point from geometry. The problem stays **linear in $c$**.
- The cosine kernel *destroys* high-frequency lighting information in shading — but **shadows and visibility re-inject it**. High-frequency evidence about the light lives in shadow geometry, not in smooth shading. Design the estimator accordingly (§4).

*Degeneracy:* $l \ge 3$ is invisible to diffuse shading — genuinely unidentifiable from matte surfaces; recoverable only via sharp shadows or specular probes.

### 2.3 Sun / directional source

$$
g_\odot(x) = V(x,\hat\omega_\odot)\,\big(\hat n \cdot \hat\omega_\odot\big)_+ , \qquad \xi = \hat\omega_\odot \in S^2 .
$$

*Evidence:* the sharpest signal in the scene — cast shadow curves (see §4.2), plus the shading-gradient field on smooth surfaces. Penumbra width gives the angular diameter: for a source of angular size $\alpha$ (sun: $\alpha \approx 9.3$ mrad), the shadow edge blur on the receiver is

$$
w \;\approx\; \alpha \cdot d_{\text{occluder}\to\text{receiver}} \qquad (\approx 9\ \text{mm per meter, for the sun}),
$$

which both *identifies* the source as solar and *calibrates* geometry scale-consistency.

### 2.4 Point / near sources

$$
g_p(x) = \frac{V(x,p)\,\big(\hat n\cdot \widehat{p{-}x}\big)_+}{4\pi\,\|p-x\|^{2}}, \qquad \xi = p \in \mathbb{R}^3 .
$$

*Evidence:* (i) **falloff curvature** — on a plane, $\log E$ under a point source is curved ($E \propto h/(h^2{+}d^2)^{3/2}$) while a directional source gives constant $E$: the second derivative of log-shading along smooth surfaces discriminates point vs directional and fits $p$. (ii) **Highlight triangulation** (§4.1). (iii) Shadow-ray triangulation as in `math.md` §5.5. Special case: **a phone torch is a point source with known $\xi$** (co-located with the tracked camera) — see §5.4.

### 2.5 Emissive scene surfaces

Not a dictionary element: any *visible* emitter is already handled by the near-field gather (§1). Compact bright emitters may additionally be promoted to a point element for the sharp direct term. Detection is physics-based (§7.6).

---

## 3. The master equation in the wild

Per channel $c$, per surface point:

$$
\boxed{\;B_c(x) \;=\; \rho_c(x)\,\Big[\, H_c(x) \;+\; \textstyle\sum_s S_{s,c}\; g_s\big(x;\, \xi_s\big) \Big] \;}
$$

Unknowns: the albedo field $\rho_c(x)$ ($3N$ numbers), the source colors $S_{s,c}$ ($3$ per source), the geometric parameters $\xi_s$ (a handful), plus per-frame exposure gains (nuisances, §7.5). Knowns: $B_c$ (measured everywhere visible), $H_c$ (gathered from measurements), all $g_s(\cdot;\xi)$ as *functions* (precomputable footprints once $\xi$ is fixed). The structure is **bilinear in $(\rho, S)$ given $\xi$**, and $\xi$ is low-dimensional — that shape dictates the whole estimation architecture (§6).

---

## 4. The evidence–parameter map: what identifies what

This is the heart of the wild problem. With $K{=}1$, every unknown must be pinned by a distinct *spatial* physical signature. The map:

| Unknown | Physical evidence | Mechanism |
|---|---|---|
| $\hat\omega_\odot$, $p$ (source geometry) | cast shadow curves | genericity: predicted shadow curve for wrong $\xi$ won't align with image discontinuities (§5.2) |
| " | specular highlights | mirror constraint / triangulation (§4.1) |
| " | log-shading curvature | $1/r^2$ vs constant falloff (§2.4) |
| source *type* (point vs sun) | penumbra width | $w \approx \alpha\, d$ (§2.3) |
| $S_{s,c}$ (source colors) | highlight chromaticity | neutral interface (Fresnel), per source |
| " | shadow-edge ratio invariant | §4.3 — albedo cancels across a shadow edge |
| " | Planck-locus band | soft prior for thermal/LED illuminants |
| ambient level vs sun level | shadow interiors | shadowed pixels obey the ambient-plus-bounce equation only — the *spatial* version of `math.md` §5.4's shadow identity: the fixed sun paints shadow/no-shadow "configurations" across space instead of across time |
| $\rho$ per-channel scale | near-field gather $H$ | $H$ is measured, fixed under the gauge (`math.md` §6.2) — and in 360° captures $H$ is *large* (ground bounce) |
| " | $\rho \le 1$; reference patch | boundary conditions |
| SH high bands ($l\!\ge\!2$) | shadowing/visibility variation | transfer vectors $T(x)$ differ where $V_{\text{env}}$ varies |

### 4.1 Highlights as source triangulators

At a specular pixel with recovered normal $\hat n$ and view direction $\hat v$, the source lies along the mirror direction

$$
\hat\omega = 2(\hat n\cdot\hat v)\,\hat n - \hat v .
$$

A far source: every highlight votes for the same $\hat\omega_\odot$. A near source: each highlight defines the ray $\{x + t\hat\omega\}$; two or more highlights (different $x$, or the same highlight observed sweeping across the surface as the camera moves) **triangulate $p$** by ray intersection. Bonus: the same pixels give the source *chromaticity* via neutral interface reflection. One capture pass, three unknowns constrained.

### 4.2 Shadows as generic identifiers

For a candidate $\xi$, the predicted shadow boundary is a specific curve $\mathcal{C}(\xi)$ on the geometry. The identifiability argument is a **transversality / genericity** statement: for the *true* $\xi^\*$, the image exhibits a consistent radiometric discontinuity along $\mathcal{C}(\xi^\*)$; for any wrong $\xi$, matching would require the albedo to contain a discontinuity *exactly along* a curve dictated by someone else's geometry — a measure-zero coincidence for generic $\rho$. This is the rigorous form of "shading follows geometry; paint doesn't." It licenses solving for $\xi$ by maximizing alignment between predicted $\mathcal{C}(\xi)$ and observed discontinuity structure — a physical estimator, not a heuristic, *because the curve family comes from the transport model.*

### 4.3 The shadow-edge ratio invariant (albedo cancels)

Take two adjacent points straddling a sun-shadow edge, close enough that $\rho$ and the slowly-varying terms match. Then per channel

$$
\frac{B_c^{\text{shadow}}}{B_c^{\text{lit}}}
= \frac{\rho_c\,\big[H_c + M_c\big]}{\rho_c\,\big[H_c + M_c + S_{\odot,c}\, g_\odot\big]}
= \frac{H_c + M_c}{H_c + M_c + S_{\odot,c}\, g_\odot} ,
$$

where $M_c$ collects the ambient/sky terms — **the albedo cancels exactly.** Consequences: (i) every sun-shadow edge in the scene carries (approximately) one shared chromatic signature, determined by the ambient-vs-sun spectral ratio — measuring it at many edges estimates that ratio robustly; (ii) an image discontinuity whose cross-edge ratio *doesn't* fit the signature is a paint edge, not a shadow — a physical edge classifier that feeds §4.2.

---

## 5. The ambiguity ledger for $K = 1$

### 5.1 The pointwise freedom — the real enemy

With a single illumination, the master equation can be satisfied by *any* light hypothesis: set $\rho_c(x) := B_c(x)/E_c(x;\theta)$ pointwise. **Lighting is unidentifiable from the pointwise equations alone.** Everything above is the escape:

- **Range:** wrong $\theta$ drives $\rho$ out of $[0,1]$ somewhere (underestimate ambient → shadow pixels demand $\rho > 1$; overestimate → bright pixels demand $\rho \approx 0$ with implausible structure).
- **Low dimensionality:** $E(\cdot;\theta)$ lives in a family of dimension $\dim\theta \ll N$, whose members have *specific geometric footprints* ($A(x)$, $T(x)$, shadow curves, $1/r^2$ fields). A wrong $\theta$ leaves *ghost structure* in $\rho = B/E(\theta)$ that is correlated with those footprints.
- **Genericity axiom (state it explicitly):** the true albedo is statistically independent of the scene's illumination-geometry fields. Formally: for generic $\rho$, coincidence of albedo structure with $\{A, T, \mathcal{C}(\xi), g_p\}$ has measure zero. This axiom is what converts "ghost structure" into an estimator: *choose $\theta$ to minimize the statistical dependence of the recovered $\rho$ on the geometric footprint fields.* It is the one non-radiometric assumption in this document — a transversality assumption, adopted knowingly.
- **The measured terms:** $H$ (and measured sky) do not rescale with hypotheses — they anchor absolute per-channel scale exactly as in `math.md` §6.2.

### 5.2 The gauge, revisited

The per-channel scale gauge $(\rho, S) \to (\alpha\rho, S/\alpha)$ survives only where **all** of these are absent: near-field bounce, visible sky, usable highlights, reference patches, and the $\rho \le 1$ bound being active. In a wild 360° scene, $H \ne 0$ almost everywhere (ground bounce) — the gauge is *better* broken in the wild than on DiLiGenT.

### 5.3 The genuinely degenerate wild scene

Far-field-only illumination, convex shadowless geometry, perfectly matte, single condition, no sky in frame: unidentifiable in principle (this is the outdoor cousin of `math.md` §6.5). Recognize it, don't fight it: instrument the capture (§5.4) or accept a one-parameter family per channel bounded by $\rho \le 1$.

### 5.4 The pocket gauge-breaker: any controlled light restores $K > 1$

A phone torch (or flash/no-flash pairs) converts the wild problem back into the configuration-differencing regime of `math.md` §5.4 — with the luxury that the source pose is *known* (camera-rigged):

$$
\Delta B_c(x) \;=\; B_c^{\text{torch on}} - B_c^{\text{torch off}} \;=\; \rho_c(x)\; S_{\text{torch},c}\; g_p\big(x;\, p_{\text{cam}}\big) \;+\; \rho_c\,\Delta H_c ,
$$

where the ambient/sun/sky terms cancel **identically** and $\Delta H$ (torch-induced bounce) is again gatherable from the difference images. One known light you carry is worth an entire identifiability section: it hands you $\rho_c$ up to the (known-hardware, sharable) torch spectrum, after which the $K{=}1$ machinery only needs to explain the *ambient* frame — a vastly easier residual problem. This is the mathematical justification for a torch-based capture method as the practical route to full de-lighting in the wild.

---

## 6. Estimation architecture

The unknown structure (bilinear in $(\rho, S)$, low-dimensional nonlinear $\xi$, measured $H$) dictates a solve order; each stage consumes the previous stage's outputs and a specific slice of evidence.

1. **Geometry & radiometry pass.** Multi-view reconstruction → surfaces, normals; per-frame exposure gains from view overlap (§7.5); per-point view-invariant (diffuse) radiance and specular residual (§7.1).
2. **Transfer precomputation.** From geometry alone: $A(x)$, SH transfer vectors $T(x)$, candidate shadow-curve family $\mathcal{C}(\xi)$, near-field gather rays (reusable across all hypotheses — geometry never changes).
3. **Nonlinear source parameters $\xi$.** From the *geometric* evidence only: highlight triangulation (§4.1), shadow alignment (§4.2), falloff curvature, penumbra width. Deliberately albedo-free — these estimators never touch $\rho$.
4. **Near-field gather.** $H_c(x)$ from measured $B$ over reconstructed surfaces + measured sky term. (If a torch is present: also $\Delta H$ from difference images.)
5. **Linear/bilinear core.** Given $\xi$: solve $(\rho, S)$ by alternating constrained least squares — $S$ linear given $\rho$; $\rho$ closed-form given $S$ (the weighted formula of `math.md` §5.4), boxed to $[0,1]$ (NNLS-style). Inject the chromatic constraints (highlight chromaticity per source, shadow-edge ratio, Planck band) as linear/soft constraints on $S$; the shared-spectrum tying across same-hardware sources reduces $S$'s dof.
6. **Model selection over the dictionary.** Add sources greedily (ambient → +SH$_1$ → +sun → +points) while the physical audits improve; stop when residuals are audit-clean (Occam via evidence, not via taste).
7. **Audits (the diagnostics are part of the math).** (a) $\rho$-range violation map → emitters or geometry error (§7.6); (b) residual correlation of $\rho$ with footprint fields $\{A, T, g_s\}$ → wrong $\theta$ (this *is* the §5.1 genericity estimator, reused as a test); (c) shadow-edge invariant consistency; (d) if $K>1$: cross-configuration $\rho$ agreement (`math.md` §5.4).

Every audit failing in a *structured* way names its own culprit — that's the practical value of deriving the estimator from transport rather than fitting a black box.

---

## 7. The toolbox: math tricks worth keeping on the bench

### 7.1 View-invariance separation (Lambertian = constant over views)

The definition of Lambertian is view-independence. Per surface point, collect its radiance across all frames that see it:

$$
L_{\text{out}}(x, \hat v_f) = \frac{B(x)}{\pi} + L_{\text{spec}}(x, \hat v_f)
\quad\Longrightarrow\quad
\widehat{B/\pi} = \operatorname{median}_f\, L_{\text{out}}(x,\hat v_f), \qquad L_{\text{spec}} = \text{positive residual}.
$$

One robust statistic splits diffuse from specular *before any lighting is solved* — the diffuse channel feeds the radiosity machinery, the specular residual feeds §4.1 triangulation and illuminant chromaticity. Multi-view capture makes this nearly free.

### 7.2 Log-domain additivity

$\log B_c = \log\rho_c + \log E_c$: shading becomes additive, and on patches where albedo is locally constant, $\nabla \log B = \nabla \log E$ — image gradients read out *lighting* gradients directly (used with the §4.3 edge classifier to know which gradients qualify). Falloff-curvature tests (§2.4) live naturally here.

### 7.3 Chromaticity projections

Project out intensity ($B_c / \sum_c B_c$) to isolate the purely chromatic constraints (highlight color, shadow-edge signature, Planck band) from the geometric ones — decouples the color sub-problem, which is 2-dimensional per source, from the $N$-dimensional spatial one.

### 7.4 Gather-ray reuse

Rays depend on geometry only: trace the cosine-weighted gather once per point, reuse the hit list for every channel, every lighting hypothesis, every torch difference, and every future relighting (`math.md` §9.2's trick, now amortized across the entire estimation loop).

### 7.5 Exposure from view overlap

Two frames $i,j$ observing the same Lambertian point under the same light: pixel ratio $= g_i/g_j$ exactly. Millions of covisible points → a log-linear graph least squares over frames pins all gains up to one global scale (absorbed into the gauge, then broken with the rest of it). Auto-exposure stops being a threat and becomes a solved sub-problem.

### 7.6 The emitter test

Where the recovered albedo insists $\hat\rho_c(x) = B_c/(E_c + H_c) > 1$ beyond noise, energy conservation is violated — the point is (generically) an **emitter**, with emission lower-bounded by $B_c - (E_c + H_c)$. Flag it, feed its measured $B$ into the gather (already automatic), optionally promote to a point-source element. Physics-based lamp detection, no classifier.

### 7.7 Temporal freebies in long captures

Illumination that drifts during capture is *signal*, not noise: passing clouds modulate the sun/ambient ratio (random natural configurations — a weak, free $K>1$); the sun's motion over a long session sweeps shadow curves (a slow, exactly-modeled $\xi(t)$). Both slot into the differencing identities unchanged.

---

## 8. Specializations

- **DiLiGenT(-MV):** $K = 96 \times$ views, calibrated $\xi$ and $S$ (with imperfect intensities) — the dictionary collapses to known directionals + weak self-$H$; the gauge is handled in hardware *if* intensity re-optimization is tied to a shared spectrum (previous discussion). The §6 architecture reduces to `math.md` §5 with the §4.1/§4.3 constraints as regularizers.
- **Torch scan (your capture):** $K = 2$ per viewpoint via on/off differencing; §5.4 is the backbone; the ambient frame is then solved with the $K{=}1$ machinery on a much smaller residual problem.
- **Pure casual capture:** the full §4–§6 program; success is scene-dependent and the audits say when the degenerate case (§5.3) has been hit.

---

## 9. Open ideation list

1. Formalize the genericity estimator of §5.1 as a proper statistic (e.g., a randomization test: correlation of $\hat\rho$ with footprint fields vs. its null distribution under footprint shuffles).
2. Uncertainty propagation end-to-end: report the gauge mode's posterior variance as a function of $\|H\|$/SNR — the honest "how broken is the gauge *here*" number.
3. Optimal capture: given a scene, which torch trajectory maximizes the information matrix of $(\rho, S, \xi)$? (Move the light a lot → made quantitative.)
4. Penumbra-based source *size* estimation as a first-class dictionary parameter (area lights between "point" and "sun").
5. Mirror-ball-free light probes: use the scene's own glossiest surface, found via §7.1's specular residual, as an improvised probe.
6. Polarization add-on (clip-on filter): optical specular/diffuse split to replace §7.1 where view sampling is thin.
7. Joint refinement: a differentiable version of the full pipeline (geometry → footprints → bilinear core) for a final polish pass — the closed-form solves as initialization, gradients only at the end.
8. Sky modeling: replace the generic SH far field with a physical sky model (sun + Rayleigh-scattered skylight) when outdoors — fewer parameters, more physics.

---

## Appendix: assumption ledger (deltas from `math.md`)

Recovered (not known) geometry with propagated error · RGB broadband sensing (spectral claims limited to 3-channel projections; Planck/locus priors soften this) · single or few illumination conditions · exposure/WB nuisances solved via §7.5 · genericity axiom (§5.1) adopted explicitly as the sole non-radiometric assumption.
