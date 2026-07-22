# Physically Exact De-Lighting: Recovering Albedo and Relighting a Scene from First Principles

*A step-by-step derivation, from radiometry to a solvable linear system, with implementation notes.*

---

## 0. Problem statement and notation

**Setup.** An opaque object sits on a table inside a room. Two light sources: a small lamp near the object (a *point source*) and diffuse room light (an *isotropic ambient* field). We photograph the scene many times, moving the lamp between shots.

**Goal.** Recover the intrinsic reflectance (albedo) $\rho(x,\lambda)$ of every surface point $x$ at every wavelength $\lambda$, and the source parameters, so that the scene can be relit under arbitrary new lighting exactly as physics would relight it.

**Notation.**

| Symbol | Meaning | Units |
|---|---|---|
| $\lambda$ | wavelength | nm |
| $\Phi(\lambda)$ | spectral power of the point lamp | W·nm⁻¹ |
| $L_a(\lambda)$ | radiance of the isotropic ambient field | W·m⁻²·sr⁻¹·nm⁻¹ |
| $L(x,\omega,\lambda)$ | spectral radiance at $x$ in direction $\omega$ | W·m⁻²·sr⁻¹·nm⁻¹ |
| $E(x,\lambda)$ | spectral irradiance (incoming power per area) | W·m⁻²·nm⁻¹ |
| $B(x,\lambda)$ | radiosity / exitance (outgoing power per area) | W·m⁻²·nm⁻¹ |
| $\rho(x,\lambda)$ | albedo, $0 \le \rho \le 1$ | — |
| $\hat n(x)$ | surface normal at $x$ | — |
| $x_L^{(k)}$ | lamp position in configuration $k = 1,\dots,K$ | m |
| $V(\cdot,\cdot)$ | binary visibility (1 if unobstructed) | — |
| $S$ | the union of all surfaces (object, table, walls) | — |

Everything is done **per wavelength**; wavelengths never mix (no fluorescence), so fix $\lambda$ and drop it when unambiguous. "Color" is just the collection of answers over $\lambda$.

---

## 1. Radiometry from first principles

### 1.1 The four radiometric quantities

Start from radiant flux $\Phi$ [W], the total power carried by light. Derived densities:

- **Radiant intensity** $I = \dfrac{d\Phi}{d\omega}$ [W·sr⁻¹] — power per solid angle (for compact sources).
- **Irradiance** $E = \dfrac{d\Phi}{dA}$ [W·m⁻²] — power arriving per unit surface area.
- **Radiance** $L = \dfrac{d^2\Phi}{dA\cos\theta \, d\omega}$ [W·m⁻²·sr⁻¹] — power per unit *projected* area per unit solid angle.

Radiance is the fundamental field: all others are integrals of it. In particular,

$$
E(x) = \int_{\Omega^+} L_{\text{in}}(x,\omega)\, (\hat n \cdot \omega)\, d\omega ,
$$

the cosine-weighted integral of incoming radiance over the upper hemisphere $\Omega^+$.

### 1.2 Radiance is conserved along rays

Take two small patches $dA_1, dA_2$ separated by $r$ in empty space. The flux through both is

$$
d^2\Phi = L \, dA_1 \cos\theta_1 \, d\omega_{1\to2} = L \, \frac{dA_1\cos\theta_1 \, dA_2\cos\theta_2}{r^2},
$$

which is symmetric in the two patches. Hence the same $L$ describes the ray at both ends: **radiance is constant along a ray** in non-absorbing media. Consequence: a focused camera pixel reports the radiance of the surface patch it images, *independent of distance*. This is the entire justification for treating photographs as physical measurements.

### 1.3 The BRDF

Reflection at a surface is defined by the bidirectional reflectance distribution function:

$$
f_r(x,\omega_i \to \omega_o) \;=\; \frac{dL_{\text{out}}(\omega_o)}{L_{\text{in}}(\omega_i)\,(\hat n\cdot\omega_i)\, d\omega_i} \quad [\text{sr}^{-1}],
$$

obeying **Helmholtz reciprocity** $f_r(\omega_i,\omega_o) = f_r(\omega_o,\omega_i)$ and **energy conservation**

$$
\int_{\Omega^+} f_r(\omega_i,\omega_o)\,(\hat n\cdot\omega_o)\, d\omega_o \;\le\; 1 \quad \text{for all } \omega_i .
$$

### 1.4 Lambertian surfaces and what "albedo" means

A Lambertian surface emits reflected radiance equally in all directions: $L_{\text{out}}$ independent of $\omega_o$. Its exitance is

$$
M = \int_{\Omega^+} L_{\text{out}} \cos\theta \, d\omega = L_{\text{out}} \int_0^{2\pi}\!\!\int_0^{\pi/2} \cos\theta \sin\theta \, d\theta\, d\varphi = \pi L_{\text{out}} .
$$

Define the albedo by $M = \rho E$ (fraction of incident power re-emitted). Then

$$
\boxed{\,L_{\text{out}} = \frac{\rho\, E}{\pi}, \qquad f_r = \frac{\rho(x,\lambda)}{\pi}, \qquad 0 \le \rho \le 1.\,}
$$

The bound $\rho \le 1$ is energy conservation; we will need it later as a boundary condition.

---

## 2. The illumination model

### 2.1 Point source

An isotropic emitter of spectral power $\Phi(\lambda)$ has intensity $I = \Phi/4\pi$. The flux it delivers to a patch $dA$ at distance $r$, tilted by $\theta$, is $d\Phi = I\, d\omega = I\, dA\cos\theta / r^2$, so

$$
E_{\text{pt}}^{(k)}(x,\lambda) \;=\; \frac{\Phi(\lambda)}{4\pi}\,\frac{V_k(x)\,\big(\hat n\cdot \hat l_k\big)_+}{r_k^2},
\qquad r_k = \|x_L^{(k)}-x\|,\;\; \hat l_k = \frac{x_L^{(k)}-x}{r_k}.
$$

$V_k(x) \in \{0,1\}$ encodes cast shadows and is pure geometry. (A physically extended lamp replaces this with $E = \int_{A_L} L_e \cos\theta\cos\theta_L\, V/r^2 \, dA_L$, which produces penumbrae; we use the point idealization.)

### 2.2 Isotropic ambient

Model the room as an isotropic radiance field: $L_{\text{in}}(x,\omega) = L_a(\lambda)$ for every unblocked direction. Then

$$
E_{\text{amb}}(x,\lambda) = L_a(\lambda)\int_{\Omega^+} V_{\text{env}}(x,\omega)\,(\hat n\cdot\omega)\, d\omega
\;=\; \pi\, L_a(\lambda)\, A(x),
$$

$$
A(x) \;\triangleq\; \frac{1}{\pi}\int_{\Omega^+} V_{\text{env}}(x,\omega)\,(\hat n\cdot\omega)\, d\omega \;\in\;[0,1].
$$

$A(x)$ — "how much of the environment this point can see" — drops straight out of the integral; on an unobstructed plane $A = 1$ since $\int \cos\theta\, d\omega = \pi$. (Computer graphics later named this quantity *ambient occlusion*; here it is simply physics.)

**Why isotropic ambient rather than a directional light?** A directional source is a delta function in direction — a point lamp pushed to infinity — and produces a second family of hard parallel shadows, not diffuse fill. The isotropic field is the correct idealization of "the room's walls as one large uniform emitter," and it integrates in closed form. Important caveat: the ambient term models the *room*; the object's own interreflections must **not** be dumped into it — they are handled exactly in §3.

### 2.3 Total direct irradiance

$$
E_d^{(k)}(x,\lambda) = E_{\text{pt}}^{(k)}(x,\lambda) + E_{\text{amb}}(x,\lambda).
$$

---

## 3. Global transport: all bounces, exactly

### 3.1 The rendering equation

For non-emissive surfaces, outgoing radiance is the BRDF-weighted integral of incoming radiance:

$$
L_{\text{out}}(x,\omega_o) = \int_{\Omega^+} f_r(x,\omega_i,\omega_o)\, L_{\text{in}}(x,\omega_i)\,(\hat n\cdot\omega_i)\, d\omega_i,
$$

where $L_{\text{in}}(x,\omega)$ is either a source term or the outgoing radiance of the first surface hit by the ray $(x,\omega)$. This recursion is the entirety of light transport.

### 3.2 Lambertian closure: the radiosity equation

Substitute $f_r = \rho/\pi$. Outgoing radiance becomes direction-independent, so a single scalar per point suffices: the radiosity $B(x) = \pi L_{\text{out}}(x)$.

Compute the irradiance contributed at $x$ by another surface point $x'$: that point emits radiance $B(x')/\pi$ toward $x$; converting the solid-angle measure to an area measure, $d\omega = \cos\theta' \, dA' / \|x-x'\|^2$, gives contribution $(B(x')/\pi)\, G(x,x')\, dA'$ with the **geometric kernel**

$$
G(x,x') = \frac{\cos\theta \, \cos\theta' \, V(x,x')}{\|x-x'\|^{2}},
$$

both cosines clamped at zero. Summing over all surfaces and adding direct light:

$$
\boxed{\;B(x,\lambda) = \rho(x,\lambda)\left[\,E_d(x,\lambda) + \frac{1}{\pi}\int_S B(x',\lambda)\, G(x,x')\, dA'\,\right].\;}
$$

This is a Fredholm integral equation of the second kind — the **radiosity equation**.

### 3.3 Operator form and the bounce series

Write $\mathrm{R}$ for pointwise multiplication by $\rho$ and $\mathrm{F}$ for the transport operator $(\mathrm{F}B)(x) = \frac{1}{\pi}\int_S B\,G\,dA'$. Then

$$
B = \mathrm{R}\,(E_d + \mathrm{F}B)
\quad\Longrightarrow\quad
B = (\mathbb{1}-\mathrm{R}\mathrm{F})^{-1}\mathrm{R}\,E_d = \sum_{b\ge 1} (\mathrm{R}\mathrm{F})^{b-1}\,\mathrm{R}\,E_d .
$$

**Convergence.** The row sums of $\mathrm{F}$ are $\frac{1}{\pi}\int_S G\, dA' = $ (fraction of the hemisphere covered by surfaces) $\le 1$, so $\|\mathrm{F}\|_\infty \le 1$ and $\|\mathrm{R}\mathrm{F}\|_\infty \le \rho_{\max} < 1$. The Neumann series converges geometrically with ratio $\rho_{\max}$: *bounces die out because surfaces absorb.*

**Spectral bookkeeping (remember this).** The $b$-th term, $(\mathrm{R}\mathrm{F})^{b-1}\mathrm{R}E_d$, applies $\rho(\cdot,\lambda)$ exactly $b$ times: light that has bounced $b$ times carries a *product of $b$ albedo spectra*. This fact will break the color ambiguity in §6.

---

## 4. The measurement model

An ideal camera is linear and, behind a narrowband filter centered at $\lambda_j$, its pixel imaging surface point $x_p$ reads

$$
I_k(p,\lambda_j) \;=\; g \cdot L_{\text{out}}(x_p \to \text{cam}, \lambda_j) \;=\; \frac{g}{\pi}\, B_k(x_p, \lambda_j),
$$

with radiometric gain $g$ known from calibration, thanks to §1.2 (radiance conservation along the viewing ray). Photographing the scene from enough viewpoints therefore yields the **data tensor**

$$
B_k(x_i,\lambda_j), \qquad k=1..K \text{ (lamp positions)},\;\; i = 1..N \text{ (surface samples, including table and walls)},\;\; j=1..\Lambda .
$$

This "brightness of every surface, in every configuration, at every wavelength" is the sole photographic input.

---

## 5. Exact inversion

### 5.1 The key identity: bounce light is computable from data

The interreflection term in §3.2 depends only on $B$ (measured) and $G$ (geometry). Define

$$
H_k(x,\lambda) \;\triangleq\; \frac{1}{\pi}\int_S B_k(x',\lambda)\, G(x,x')\, dA' \qquad \textbf{(computable — no unknowns).}
$$

*Intuition:* to know how much the table glows onto the object, you do not need to know why the table is bright — you photographed how bright it is. This single move converts "infinitely many unknown bounces" into a known quantity.

### 5.2 The master equation

Substituting §2 into §3.2, each surface point, configuration, and wavelength obeys

$$
\boxed{\;B_k(x,\lambda) \;=\; \rho(x,\lambda)\,\Big[\underbrace{\Phi(\lambda)\,a_k(x)}_{\text{lamp}} \;+\; \underbrace{\pi L_a(\lambda)\, A(x)}_{\text{ambient}} \;+\; \underbrace{H_k(x,\lambda)}_{\text{bounces}}\Big],\;}
\qquad
a_k(x) = \frac{V_k(x)\,(\hat n\cdot\hat l_k)_+}{4\pi\, r_k^{2}} .
$$

Knowns: $B_k$, $a_k$, $A$, $H_k$. Unknowns per wavelength: the field $\rho(x)$ and two scalars $\Phi, L_a$.

### 5.3 The problem is linear

Introduce $\alpha(x) \triangleq \rho(x)\Phi$, $\beta(x) \triangleq \rho(x)\,\pi L_a$. The master equation becomes, at each point $x$,

$$
B_k \;=\; \alpha\, a_k(x) \;+\; \beta\, A(x) \;+\; \rho\, H_k(x), \qquad k = 1,\dots,K,
$$

which is **linear in the three per-point unknowns** $(\alpha, \beta, \rho)$. With $K \ge 3$ lamp positions (so that the vectors $\{a_k\}$ and $\{H_k\}$ vary independently across $k$ — moving the lamp changes both the direct pattern and the bounce field), this is an overdetermined linear least-squares problem per point:

$$
\begin{pmatrix} a_1 & A & H_1 \\ a_2 & A & H_2 \\ \vdots & \vdots & \vdots \\ a_K & A & H_K \end{pmatrix}
\begin{pmatrix} \alpha \\ \beta \\ \rho \end{pmatrix}
=
\begin{pmatrix} B_1 \\ B_2 \\ \vdots \\ B_K \end{pmatrix}.
$$

Then $\rho(x,\lambda)$ is read off directly, and the global light parameters are recovered as $\Phi = \alpha/\rho$, $\pi L_a = \beta/\rho$ — which must agree across all points $x$, a massive built-in consistency check. (In practice, enforce global consistency by a second pass: fix $\Phi, L_a$ to their robust global estimates, then recompute $\rho$; see §5.5.)

**Where the famous ambiguity lives.** If interreflections vanish ($H_k \equiv 0$), the third column disappears and the system has rank 2: only the *products* $\rho\Phi$ and $\rho L_a$ are determined, not the factors. The albedo–illuminant ambiguity of §6 is precisely this rank deficiency — and the $H$ column is what removes it.

### 5.4 Useful sub-identities

- **Shadow pixels.** Where $V_k = 0$: $\;B_k = \rho\,(\pi L_a A + H_k)$ — a pure ambient-plus-bounce equation. Shadows separate the two sources for free.
- **Configuration differencing.** $\;B_k - B_{k'} = \rho\,\big[\Phi\,(a_k - a_{k'}) + H_k - H_{k'}\big]$ — the ambient term cancels *identically*, because only the lamp moved.
- **Weighted albedo estimate.** Once $\Phi, L_a$ are fixed, the denominator $D_k = \Phi a_k + \pi L_a A + H_k$ is known and the least-squares albedo over all configurations is
$$
\rho(x,\lambda) = \frac{\sum_k B_k D_k}{\sum_k D_k^2},
$$
which is more noise-robust than any single division $B_k/D_k$.

### 5.5 If the lamp positions are unknown

Three physical routes, in increasing generality:

1. **Shadow triangulation.** Every umbra boundary point $x_s$ lies on the line through the lamp and a silhouette point $x_o$ of the occluder. Two independent (boundary, silhouette) pairs give two lines whose intersection is $x_L$.
2. **Falloff on a known plane.** On the empty table (a plane), with lamp height $h$ above foot point $x_\perp$:
$$
E_{\text{pt}}(x) = \frac{\Phi}{4\pi}\,\frac{h}{\big(h^2 + \|x - x_\perp\|^2\big)^{3/2}},
$$
a two-parameter falloff profile fit.
3. **Joint solve.** Treat $\{x_L^{(k)}\}$ as $3K$ additional unknowns; the system of §5.3 becomes mildly nonlinear (through $a_k$) but remains finitely parameterized and heavily overdetermined.

---

## 6. Identifiability: breaking the albedo–illuminant color ambiguity

### 6.1 The gauge symmetry

Restricting to *single-bounce* physics, the prediction at each $\lambda$ is $\rho\,(\Phi a + \pi L_a A)$, invariant under

$$
\rho \to \alpha(\lambda)\,\rho, \qquad (\Phi, L_a) \to \big(\Phi, L_a\big)/\alpha(\lambda)
$$

for any positive function $\alpha(\lambda)$: *"red wall or red light"* is undecidable from direct light alone. Physics breaks the gauge three independent ways.

### 6.2 Break #1 — interreflection (the main one)

$H_k$ is **measured**, hence fixed under the gauge transformation. The transformed prediction is

$$
B' = \alpha\rho\left(\frac{\Phi a + \pi L_a A}{\alpha} + H\right) = \rho\,(\Phi a + \pi L_a A) + \alpha\,\rho H,
$$

which equals the true $B = \rho(\Phi a + \pi L_a A) + \rho H$ **iff $\alpha = 1$**, wherever $\rho H \ne 0$. *Any point receiving bounce light pins the gauge.* Equivalently (§3.3): $b$-times-bounced light has spectrum $\propto \Phi(\lambda)\,\rho_1(\lambda)\cdots\rho_b(\lambda)$ — it is "dyed" once per bounce, and comparing once-dyed to twice-dyed light factors paint from illuminant.

**Worked closed form (two facing patches).** Patches 1, 2 with mutual couplings $F_{12}, F_{21}$ and direct irradiances $E_1, E_2$:

$$
B_1 = \rho_1\,(E_1 + F_{12} B_2), \quad B_2 = \rho_2\,(E_2 + F_{21} B_1)
\;\Longrightarrow\;
B_1 = \rho_1\,\frac{E_1 + \rho_2 F_{12} E_2}{1 - \rho_1\rho_2 F_{12} F_{21}} .
$$

The response is *nonlinear* in the albedos — measurements at two lamp positions (changing the ratio $E_1{:}E_2$) determine $\rho_1, \rho_2$ **absolutely**, not just up to scale.

### 6.3 Break #2 — Fresnel highlights

A dielectric surface reflects via two mechanisms: *body* reflection (light enters, scatters among pigment, exits — this is the Lambertian, albedo-tinted part) and *interface* reflection (the specular lobe), governed by the Fresnel equations through the refractive index $n(\lambda)$. Across the visible band, dispersion is tiny (e.g., crown glass: $n \approx 1.514 \to 1.522$), so the Fresnel reflectance is nearly wavelength-flat and

$$
L_{\text{specular}}(\lambda) \;\propto\; \Phi(\lambda):
$$

**a highlight is a free spectrometer for the lamp.** Bonus from wave optics: interface reflection is partially polarized (fully, at Brewster's angle $\theta_B = \arctan n \approx 56°$) while body reflection is depolarized — so a rotating polarizer separates the two components physically before any computation.

### 6.4 Break #3 — the Planckian prior

Thermal illuminants ("shades of white/yellow") lie on the blackbody locus:

$$
\Phi(\lambda; T) \;\propto\; \lambda^{-5}\Big(e^{\,c_2/\lambda T} - 1\Big)^{-1}, \qquad c_2 = \frac{hc}{k_B} \approx 1.4388\times 10^{-2}\ \text{m·K}.
$$

One temperature $T$ (plus a scale) replaces an arbitrary spectrum — collapsing the per-$\lambda$ gauge to two numbers. Use as a cross-check on #1–#2, or as the rescue when only broadband RGB data exist.

### 6.5 The genuinely degenerate case

A convex, perfectly matte object in a black room: $H \equiv 0$, no highlights. Then the per-$\lambda$ scale is *physically unknowable* — "dim white paint" and "bright gray paint under a stronger lamp" emit literally identical photons. Resolve by a boundary condition: the energy bound $\rho(\lambda) \le 1$ (gives an interval / upper normalization), or one reference patch of known reflectance (e.g., a ≈99% white standard). This is not a hack; it is the required boundary condition of an otherwise scale-invariant problem.

---

## 7. Relighting

With $\rho(x,\lambda)$ and the geometry (hence $\mathrm{F}$) recovered, any new lighting $E_d^{\text{new}}$ — new lamps, colored lights, an environment map — is rendered by solving the same equation forward:

$$
B^{\text{new}} = (\mathbb{1}-\mathrm{R}\mathrm{F})^{-1}\,\mathrm{R}\,E_d^{\text{new}},
\qquad\text{via the fixed point}\qquad
B^{(t+1)} = \mathrm{R}\,\big(E_d^{\text{new}} + \mathrm{F}\,B^{(t)}\big),
$$

whose error contracts by $\rho_{\max}$ per iteration ($\sim \log\varepsilon / \log\rho_{\max}$ passes, i.e., 10–30 in practice). This is *literally the equation the room itself solves* when you flip the switch — same operator, new source term — so shadows, bounces, and color bleeding in the relit result match the physical experiment by construction.

---

## 8. Assumptions, and what breaks when you relax them

- **Lambertian → general BRDF.** Outgoing radiance becomes direction-dependent; the unknown is $f_r(x,\omega_i,\omega_o,\lambda)$ and the state variable is $L(x,\omega)$, not a scalar $B(x)$. Everything in §3–§5 generalizes but requires dense sampling in *both* view and light (a reflectance-field measurement).
- **Fluorescence.** Wavelengths couple; $\rho(\lambda)$ becomes a re-radiation matrix $\rho(\lambda_{\text{in}} \to \lambda_{\text{out}})$ (Donaldson matrix). Solvable with narrowband *illumination* scanning.
- **Subsurface scattering.** Reflection becomes nonlocal in $x$: the BSSRDF $S(x_i,\omega_i; x_o,\omega_o)$.
- **Participating media (smoke, water).** Radiance is no longer conserved along rays; replace §1.2 with the radiative transfer equation ($\sigma_a$, $\sigma_s$, phase function).
- **Geometry uncertainty.** Errors in $\hat n$ and $r$ propagate directly into $a_k$ and $G$, hence into $\rho$; geometry must be measured to the accuracy you want in the albedo.

---

## 9. Computational implementation

### 9.1 Discretization and data sizes

Sample the surfaces at $N$ points (mesh vertices or texels) with area weights $w_i$ and normals $\hat n_i$. Representative scale: $N = 5\times10^5$ samples, $K = 32$ lamp positions, $\Lambda = 31$ spectral bands (400–700 nm at 10 nm). Data tensor $B$: $K \cdot N \cdot \Lambda \approx 5\times10^8$ floats $\approx$ 2 GB in fp32 — stream per wavelength.

### 9.2 Never materialize $G$

The kernel $G$ has $N^2 = 2.5\times10^{11}$ entries. Do **not** build it. Compute $H_k$ matrix-free by a Monte Carlo *gather*:

$$
\pi L_a A(x) + H_k(x) \;=\; \int_{\Omega^+} L_{\text{in}}(x,\omega)\cos\theta\, d\omega
\;\approx\; \frac{1}{M}\sum_{m=1}^{M}
\begin{cases}
B_k(\text{hit}(x,\omega_m)) & \text{ray hits a surface} \\
\pi L_a & \text{ray escapes to the environment}
\end{cases}
$$

with directions $\omega_m$ drawn from the cosine-weighted density $p(\omega) = \cos\theta/\pi$ (the $\pi$'s and cosines cancel exactly — that is why the estimator is a plain average). Three bonuses:

1. **One ray pass yields $A$, $H_k$, and their sum simultaneously** ($A$ = the fraction of escaping rays).
2. **Rays are wavelength-independent**: trace once per $(x, k)$, reuse the same hit points for all $\Lambda$ bands. This is a $\Lambda\times$ saving.
3. Ray casting is exactly what GPU RT hardware accelerates. Budget: $K \cdot N \cdot M$ rays with $M = 256$ gather rays $\approx 4\times10^9$ rays — seconds to minutes on an RTX-class GPU.

### 9.3 Geometry pass

$a_k(x)$ needs one shadow ray per $(x,k)$: $K\cdot N = 1.6\times10^7$ rays — negligible. Normals and areas come from the scan. Precompute and cache $a_k$, $A$ once.

### 9.4 The per-point linear solve

Each point solves a $K \times 3$ least-squares system (§5.3): trivially parallel over $N\cdot\Lambda$ points — one small QR per thread. Then robustly aggregate $\Phi = \text{median}_x(\alpha/\rho)$, $L_a = \text{median}_x(\beta/\pi\rho)$; optionally alternate (fix $\Phi, L_a \Rightarrow \rho$ closed form via §5.4; fix $\rho \Rightarrow (\Phi,L_a)$ linear) — this alternating least squares converges monotonically.

### 9.5 Conditioning and noise — the actual failure modes

- **Move the lamp a lot** (vary distance *and* direction). If all $a_k$ are near-proportional, the $K\times3$ matrix is rank-deficient and $(\Phi, L_a)$ blow up. Include configurations that put many pixels in and out of shadow.
- **Mask small denominators.** The final division amplifies noise where $D_k$ is small (deep shadow, grazing incidence). Weight by $D_k^2$ (§5.4 does this automatically) and mask pixels below an irradiance floor.
- **Sensor linearity is non-negotiable.** Shoot RAW; bracket exposures into HDR; calibrate gain, vignetting, and (if RGB) the spectral response. Any nonlinearity lands directly in $\rho$.
- **Coverage holes.** $H$ needs $B$ on *all* surfaces; occluded wall patches leave holes. Add viewpoints, or accept a bounded error term for the unseen fraction.

### 9.6 Sensible optimizations

- **Two-resolution transport.** Bounce light is spatially low-frequency (the integral blurs). Compute $H$ on a coarse sampling ($N_c \sim 10^4$) and interpolate; keep $\rho$ at full resolution. Orders-of-magnitude savings at negligible error.
- **Ray reuse across $\lambda$ and across the estimator** (§9.2) — the largest single win.
- **Low-discrepancy sampling** (blue-noise / Sobol directions) for the gather: error $\sim M^{-1}$ instead of $M^{-1/2}$ on smooth integrands.
- **Reciprocity.** $G(x,x') = G(x',x)$: if you do materialize a coarse transport matrix, it is symmetric — halve the work, and reuse it unchanged for every wavelength, configuration, and every future relighting.
- **Wavelength parallelism.** Bands never interact: trivially distribute across GPU streams.
- **Warm starts from shadow pixels** (§5.4) for $L_a$, and from the differencing identity for $\Phi$.
- **Choose the illuminant to be known.** If the lamp is a set of narrowband LEDs, $\Phi(\lambda)$ is known *by construction* and the color ambiguity never arises. Cheapest possible fix — solve the problem in hardware.
- **Relighting loop:** Jacobi iterations = repeated gather passes over the current $B$; 10–30 passes, each a single GPU kernel.

### 9.7 Verdict

Nothing is impossible. The two genuinely expensive pieces — visibility ray casting and the bounce gather — are precisely the operations modern GPUs were built for; the two genuinely hard pieces are experimental (full-surface multi-view coverage, radiometric/spectral calibration), not computational. A single good GPU handles the scales above comfortably. The only *information-theoretic* limit is spectral hardware: a plain RGB camera yields a 3-channel projection of $\rho(\lambda)$ (metamerism), so full spectra require narrowband filters or narrowband illumination.

---

## 10. Other ideas that attack the same problem

*(The main, exact route is the transport inversion of §5. These are alternatives, special cases, or complements — most are corollaries of the same physics.)*

1. **Differentiable / Monte Carlo inverse rendering.** Solve the same equations by gradient descent through a stochastic path-tracing estimator of §3.1. Same physics; trades the closed-form structure of §5 for generality (arbitrary BRDFs, unknown geometry).
2. **Lambertian photometric stereo.** Ignore bounces and ambient, use $\ge 3$ distant lights: $I_k = \rho\,\hat n\cdot \hat l_k$ is linear in $\rho\hat n$ per pixel — recovers *normals and albedo* simultaneously. The classic single-bounce special case of §5.3.
3. **One-light-at-a-time capture (light stage / reflectance fields).** Light transport is linear in the sources, so measuring the image under each basis light gives the full light→image operator; relighting is a matrix multiply. Model-free, data-hungry, no albedo factorization.
4. **Interreflection bootstrapping.** Iterate: assume no bounces → estimate $\rho$ → predict $H$ from the estimate → subtract → re-estimate. Converges to the same fixed point as §5 (it is Neumann iteration on the inverse problem); simplest to implement.
5. **Helmholtz stereopsis.** Swap the camera and the lamp between two shots; reciprocity ($f_r$ symmetric) makes the *unknown BRDF cancel* in the pair, yielding geometry/reflectance constraints valid for arbitrary materials. Pure physics, wonderfully clever.
6. **Polarization separation.** Use Fresnel polarization (§6.3) to strip specular from diffuse optically before any solving; also yields the illuminant spectrum from the specular channel.
7. **Spectral multiplexing.** Scan narrowband illumination (LEDs / monochromator) rather than narrowband sensing — makes $\Phi(\lambda)$ known and handles fluorescence (measures the full Donaldson matrix).
8. **Planck-locus constrained solve.** When only RGB data exist, parameterize the illuminant by temperature (§6.4) and fit — the physically-grounded version of "assume white/yellow light."
9. **Flash / no-flash differencing.** A two-configuration instance of §5.4: the difference image is a single-known-light photograph with ambient exactly cancelled.
10. **Low-order environment expansion.** Expand incident light in spherical harmonics; Lambertian reflection is a spherical convolution that low-passes illumination to ~9 coefficients — a compact, physically exact model for smooth environment light (though it cannot represent the nearby point lamp well).
11. **Heuristic intrinsic decomposition (Retinex-style).** Assume shading varies smoothly and albedo is piecewise-constant, split the image accordingly. *Not* physics-exact — listed as the contrast: it is what one does when none of the measurements above are available.

---

## Appendix: assumption ledger

Opaque Lambertian surfaces · no fluorescence or subsurface scattering · non-participating air · geometry known · linear, radiometrically calibrated, narrowband sensing · static scene during capture.
