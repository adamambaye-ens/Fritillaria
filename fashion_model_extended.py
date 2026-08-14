"""
fashion_model_extended.py
==========================
"Large domain" variant of fashion_model.py: the same three-phase model for
the checkerboard pattern of the snake's head fritillary (Fritillaria
meleagris), but with n_nervures = 15 instead of 7 (same pixel resolution
per vein/spot, larger physical domain). Does not modify fashion_model.py.

Model architecture: identical to fashion_model.py (see that file for the
full description of phases A/B/C and of the highres_factor parameter).

Differences from fashion_model.py
------------------------------------
Going from 7 to 15 veins while keeping width_base/height_base=50 (i.e.
the same physical domain) breaks the pattern: Gray-Scott does not have
time to propagate the pattern up to the top of a wider domain from a
single row of seeds, and some inter-vein compartments end up competing
for substrate and stay empty, or merge into a single large blob ("stripe"
instead of a column of distinct spots). An automated search (~200 trial
runs, low resolution then confirmed in HD) was used to recover a stable
parameter set:

  1. width_base = height_base = 115 (instead of 50x50): the domain grows
     in both dimensions to keep the inter-vein spacing identical to the
     7-vein reference ((115-1)/14 ~= (50-1)/6 ~= 8.14 px) AND remain a
     square with more rows of spots, not just more columns.
  2. n_steps = 138,000 (instead of 60,000), scaled proportionally to
     height_base/height_base_ref, to give the spot-replication front
     enough time to cross the whole enlarged domain.
  3. Seeds on 2 rows -- one at the bottom of the domain (petiole), one
     at the top -- instead of a single row at the bottom. This is NOT a
     convenience choice: with a single origin (bottom only), the spot-
     replication front gets durably stuck in at least 1 out of 14
     compartments, regardless of the F/k/nervure_strength/Du/Dv/n_steps
     combination tried (exhaustive search: F alone, Du/Dv, initial
     concentrations seed_U/seed_V, n_steps up to x4, highres_factor up
     to 4 -- none eliminates this blockage, which is a robust
     "propagation failure" phenomenon in this regime, not simply a lack
     of time or poor tuning). The two origins halve the distance each
     front has to travel and are enough to eliminate the blockage
     entirely. Explicit biological trade-off: two innervation points
     (base + tip of the tepal) rather than a single one at the petiole
     -- to be acknowledged and justified in the paper if this version is
     used for the final statistics.
  4. Phase B: source threshold PER inter-vein compartment (90% of the
     local V maximum within each compartment) instead of a global
     threshold (90% of the maximum over the whole domain). With only 6
     compartments (7 veins), the amplitude gap between neighboring spots
     stays under the 90%-of-global-max bar most of the time, so it
     worked by chance; with 14 competing compartments, the amplitude of
     the V peaks varies enough that a global threshold leaves several
     compartments with no colorant source at all, hence no pigment, even
     though phase A did form a spot there. This is the main cause of the
     "holes" observed when scaling up -- not an F/k instability.
  5. Phase B: the source search explicitly excludes the vein band
     itself. Without this exclusion, a V peak that forms right on a vein
     gets anchored as a source, then gets wiped out by the impermeable
     mask in phase B, and the corresponding spot ends up split in two or
     smothered.
  6. nervure_strength = 0.62 (instead of 0.4) and nervure_width_px = 1
     (instead of 2) -- 0.4/2 was NOT optimal at this scale despite what
     an initial automated search indicated: with the corrected phase B
     threshold (points 4-5), up to 68% of spots still formed directly on
     a vein (diffusion only reduced by 40%, not blocked). A joint search
     (nervure_strength x F x nervure_width_px, with the two-row seeding
     from point 3) was run for a compromise that eliminates both the
     vein/spot overlap AND the compartments that degenerate into a
     continuous stripe instead of a column of distinct spots.
  F and k stay identical to the 7-vein reference (0.0382 / 0.065) --
  only their interaction with nervure_strength/width/seeding had to
  change at this scale.
  7. Phase B: Dy = 0.15 (instead of 0.1) -- a REVISED BIOLOGICAL
     HYPOTHESIS, not a technical tweak: Dy encodes the diffusion
     anisotropy of the colorant factor along the veins (Dx:Dy = 10:1 ->
     1:0.15). Motivated by the quantification (Square Index, see
     `quantification.py` in this project's analysis code, not included
     in this repository): SI stayed close to the circle value (~0.78,
     vs 1 for a perfect square) because AR_bbox (the width/height ratio
     of the patches) plateaued around 0.87-0.91 while Fill_Ratio was
     already good (~0.88-0.91) -- the patches were "full" enough but
     wider than tall, with width bounded by the inter-vein spacing and
     height bounded by nothing. A sweep of Dy (0.05 to 0.5) on the
     already-computed V field, replaying only phases B/C and measuring
     SI/Fill_Ratio/AR_bbox/patch count through the actual
     `quantification.segment_and_measure` pipeline (Otsu threshold, not
     an ad hoc one), showed a clear plateau between Dy=0.15 and 0.18 (SI
     0.83-0.85, AR_bbox up to ~0.94, patch count stable at ~143 vs ~147
     at Dy=0.1), then progressive degradation, then collapse through
     vertical merging beyond Dy~0.3 (patch count drops from 143 to 119
     then 20). Dy=0.15 was chosen at the start of the plateau (safety
     margin before degradation) rather than at the exact peak (Dy=0.18)
     to stay robust.

Key parameter: highres_factor -- see fashion_model.py.

Measured runtime: on the order of several hours (this is NOT the fast
classroom script -- see fashion_model.py and turing_pattern_vein_free.py
for that).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*args, **kwargs):
        return lambda f: f

#%%
# =============================================================================
# BASE PARAMETERS
# =============================================================================
height_base = 115         # domain height at the reference scale (pixels)
width_base  = 115         # domain width at the reference scale (pixels)
n_nervures  = 15          # number of vertical veins

highres_factor = 3        # identical to fashion_model.py: 1=115x115 | 3=345x345

n_steps = int(138e3)      # 60e3 * height_base/50: same physically simulated
                           # duration as the reference, scaled to the larger domain
dt_base = 1.0
dx_base = 1.0

Du = 0.14
Dv = 0.06
F  = 0.0382
k  = 0.065

nervure_strength    = 0.62  # see docstring, point 6
nervure_width_px    = 1
nervure_color_width = 1

seed_rows = 2               # one row at the bottom, one at the top (see docstring, point 3)

yellow    = np.array([1.0,  1.0,  0.0])
black     = np.array([0.0,  0.0,  0.0])
gray      = np.array([0.45, 0.45, 0.45])
nervures_alpha = 0.25

# Per-phase background ramps: one distinct color ramp per phase instead
# of a single violet->white gradient shared by all three, so each phase
# reads as visually distinct. Phase A and B run light(low value) ->
# dark(high value); Phase C is reversed, dark/violet (low value, majority
# background) -> light (high value, minority patches), since the real
# flower is a violet background with white patches. Phase A's green is a
# generic placeholder (not photo-sampled); Phase B's raspberry is a first
# guess, not fully settled -- only Phase C's magenta-violet is
# photo-sampled from a real flower.
PHASE_A_LIGHT = np.array([1.000, 1.000, 1.000])   # white
PHASE_A_DARK  = np.array([0.290, 0.561, 0.227])   # bud green (#4a8f3a)
PHASE_B_LIGHT = np.array([0.988, 0.906, 0.918])   # pale pink (#fce7ea)
PHASE_B_DARK  = np.array([0.788, 0.094, 0.290])   # raspberry (#c9184a)
PHASE_C_LIGHT = np.array([0.988, 0.906, 0.918])   # pale pink-white (#fce7ea)
PHASE_C_DARK  = np.array([0.639, 0.275, 0.478])   # magenta-violet (#a3467a)

#%%
# =============================================================================
# PHYSICAL SCALING
# =============================================================================
height = height_base * highres_factor
width  = width_base  * highres_factor

dx = dx_base / highres_factor
dt = dt_base / highres_factor**2

nervure_width_px_scaled    = max(1, nervure_width_px    * highres_factor)
nervure_color_width_scaled = max(1, nervure_color_width * highres_factor)
seed_radius = max(1, highres_factor - 1)

print(f"Phase A grid: {width}x{height} px  |  dx={dx:.4f}  dt={dt:.6f}  n_steps={n_steps}")

#%%
# =============================================================================
# PHASE A: TURING REACTION-DIFFUSION  (Gray-Scott model)
# =============================================================================
U = np.ones((height, width))
V = np.zeros((height, width))

nervure_positions = np.linspace(0, width - 1, n_nervures, dtype=int)

Du_map = Du * np.ones_like(U)
Dv_map = Dv * np.ones_like(V)
for x_pos in nervure_positions:
    s = max(0, x_pos - nervure_width_px_scaled // 2)
    e = min(width, x_pos + nervure_width_px_scaled // 2)
    Du_map[:, s:e] *= (1 - nervure_strength)
    Dv_map[:, s:e] *= (1 - nervure_strength)

# Seeds: one row at the very bottom (y=1, petiole) and one at the very top
# (y=height-2), one seed per inter-vein compartment and per row (see
# docstring, point 3 -- with a single origin, the front gets durably stuck
# in at least one compartment, whatever F/k/nervure_strength/Du/Dv/n_steps
# combination is used; the two origins eliminate this blockage by halving
# the propagation distance).
init_points = []
y_centers = (height - 2, 1) if seed_rows == 2 else [height - 2]
for y_center in y_centers:
    for i in range(n_nervures - 1):
        x_center = (nervure_positions[i] + nervure_positions[i + 1]) // 2
        for dy in range(-seed_radius, seed_radius + 1):
            for dx_ in range(-seed_radius, seed_radius + 1):
                yy = np.clip(y_center + dy, 0, height - 1)
                xx = np.clip(x_center + dx_, 0, width  - 1)
                U[yy, xx] = 2.0
                V[yy, xx] = 0.75
        init_points.append((y_center, x_center))

inv_dx2 = 1.0 / dx**2

def laplacian_gs(Z):
    """5-point Laplacian with periodic (wrap) boundaries."""
    Zp = np.pad(Z, 1, mode='wrap')
    return (-4*Z + Zp[2:,1:-1] + Zp[:-2,1:-1] + Zp[1:-1,2:] + Zp[1:-1,:-2]) * inv_dx2

print("Phase A: Gray-Scott simulation in progress...")
report_every = max(1, n_steps // 50)
bar_width_console = 40

for step in range(n_steps):
    Lu  = laplacian_gs(U)
    Lv  = laplacian_gs(V)
    UVV = U * V**2
    U  += dt * (Du_map * Lu - UVV + F * (1 - U))
    V  += dt * (Dv_map * Lv + UVV - (F + k) * V)

    for Z in (U, V):
        Z[0,:]  = Z[1,:];  Z[-1,:] = Z[-2,:]
        Z[:,0]  = Z[:,1];  Z[:,-1] = Z[:,-2]

    if step % report_every == 0 or step == n_steps - 1:
        pct  = (step + 1) / n_steps
        done = int(bar_width_console * pct)
        bar  = "█" * done + "░" * (bar_width_console - done)
        sys.stdout.write(f"\r  [{bar}] {100*pct:5.1f}%  step {step+1}/{n_steps}")
        sys.stdout.flush()

print("\nPhase A done.")

if np.isnan(U).any() or np.isnan(V).any():
    raise ValueError("Numerical instability detected: NaN in U or V. "
                     "Check the parameters (dt too large, or Du/Dv too high).")

# Divide by V.max() first: Gray-Scott's V never actually reaches 1 in this
# regime (typically peaks ~0.4-0.5), so np.clip(V, 0, 1) alone was a no-op
# that left the pattern permanently under-saturated.
Vn   = np.clip(V / max(V.max(), 1e-9), 0, 1)
imgA = PHASE_A_LIGHT * (1 - Vn[..., None]) + PHASE_A_DARK * Vn[..., None]
for xp in nervure_positions:
    s = max(0, int(xp - nervure_color_width_scaled // 2))
    e = min(width, int(xp + nervure_color_width_scaled // 2 + 1))
    imgA[:, s:e, :] = black
imgA_with_anchors = imgA.copy()
for (y, x) in init_points:
    r = seed_radius
    for dy in range(-r, r + 1):
        for dx_ in range(-r, r + 1):
            yy = np.clip(y + dy, 0, height - 1)
            xx = np.clip(x + dx_, 0, width  - 1)
            imgA_with_anchors[yy, xx] = yellow

#%%
# =============================================================================
# PHASES B & C
# =============================================================================
height_hr              = height
width_hr                = width
nervure_positions_hr    = nervure_positions
nervure_width_px_hr     = nervure_width_px_scaled
nervure_color_width_hr  = nervure_color_width_scaled
V_hr                    = V.copy()

#%%
# =============================================================================
# PHASE B: ANISOTROPIC DIFFUSION OF THE COLORANT FACTOR
# =============================================================================
# Source threshold PER inter-vein compartment (see docstring, point 4): 90%
# of the V maximum *within that compartment*, not of the maximum over the
# whole domain. The vein band itself is excluded from the local-maximum
# search (see docstring, point 5) so a source is never anchored on a pixel
# that will be wiped out by the impermeable mask anyway.
vein_band_mask = np.zeros((height_hr, width_hr), dtype=bool)
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_width_px_hr // 2))
    e = min(width_hr, int(xp + nervure_width_px_hr // 2 + 1))
    vein_band_mask[:, s:e] = True

nerv_sorted = np.sort(nervure_positions_hr)
sources = np.zeros((height_hr, width_hr), dtype=bool)
for i in range(len(nerv_sorted) - 1):
    cs = int(nerv_sorted[i])
    ce = int(nerv_sorted[i + 1]) + 1
    tissue = ~vein_band_mask[:, cs:ce]
    block  = V_hr[:, cs:ce]
    if tissue.any():
        local_max = block[tissue].max()
        if local_max > 0:
            sources[:, cs:ce] = (block >= 0.9 * local_max) & tissue

colorant = np.zeros((height_hr, width_hr), dtype=float)
colorant[sources] = 1.0

Dx = 1.0
Dy = 0.15  # see docstring, point 7 (revised biological hypothesis: colorant
           # diffusion anisotropy, Dx:Dy = 1:0.15 instead of 1:0.1)

permeable_mask = np.ones((height_hr, width_hr), dtype=bool)
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_width_px_hr // 2))
    e = min(width_hr, int(xp + nervure_width_px_hr // 2 + 1))
    permeable_mask[:, s:e] = False

n_iter   = 350 * highres_factor
dt_color = 0.05

print("Phase B: colorant factor diffusion...")
report_every_B = max(1, n_iter // 50)

for it in range(n_iter):
    Cx = (np.roll(colorant, -1, axis=1) + np.roll(colorant, 1, axis=1) - 2*colorant) * Dx
    Cy = (np.roll(colorant, -1, axis=0) + np.roll(colorant, 1, axis=0) - 2*colorant) * Dy
    colorant += dt_color * (Cx + Cy)
    colorant[~permeable_mask] = 0.0
    colorant[sources] = 1.0
    colorant = np.clip(colorant, 0.0, 1.0)

    if it % report_every_B == 0 or it == n_iter - 1:
        pct  = (it + 1) / n_iter
        done = int(bar_width_console * pct)
        bar  = "█" * done + "░" * (bar_width_console - done)
        sys.stdout.write(f"\r  [{bar}] {100*pct:5.1f}%  it {it+1}/{n_iter}")
        sys.stdout.flush()

print("\nPhase B done.")

imgB = PHASE_B_LIGHT[None,None,:] * (1 - colorant[...,None]) + PHASE_B_DARK[None,None,:] * colorant[...,None]
imgB[sources] = yellow
alpha = nervures_alpha
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_color_width_hr // 2))
    e = min(width_hr, int(xp + nervure_color_width_hr // 2 + 1))
    imgB[:, s:e, :] = (1 - alpha) * imgB[:, s:e, :] + alpha * gray

#%%
# =============================================================================
# PHASE C: BISTABLE COLORING (HILL FUNCTION)
# =============================================================================
n_Hill = 3.0
K_Hill = 0.05

hill_response = (colorant**n_Hill) / (colorant**n_Hill + K_Hill**n_Hill)

for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_color_width_hr // 2))
    e = min(width_hr, int(xp + nervure_color_width_hr // 2 + 1))
    hill_response[:, s:e] = 0.0

imgC = PHASE_C_DARK[None,None,:] * (1 - hill_response[...,None]) + PHASE_C_LIGHT[None,None,:] * hill_response[...,None]
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_color_width_hr // 2))
    e = min(width_hr, int(xp + nervure_color_width_hr // 2 + 1))
    imgC[:, s:e, :] = (1 - alpha) * imgC[:, s:e, :] + alpha * black

#%%
# =============================================================================
# FINAL DISPLAY: 3 images (row 1) + 3 3D surfaces (row 2)
# =============================================================================
X,    Y    = np.meshgrid(np.arange(width),    np.arange(height))
X_hr, Y_hr = np.meshgrid(np.arange(width_hr), np.arange(height_hr))

fig = plt.figure(figsize=(18, 12))
title_fontsize = 24
ratio_z  = 0.15

# 3D surfaces use the same per-phase ramps as the flat 2D images above,
# so each phase reads as the same "color story" in both rows.
from matplotlib.colors import LinearSegmentedColormap
cmap_A = LinearSegmentedColormap.from_list("phaseA", [PHASE_A_LIGHT, PHASE_A_DARK])
cmap_B = LinearSegmentedColormap.from_list("phaseB", [PHASE_B_LIGHT, PHASE_B_DARK])
cmap_C = LinearSegmentedColormap.from_list("phaseC", [PHASE_C_DARK, PHASE_C_LIGHT])

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(imgA_with_anchors, origin='lower', aspect='auto')
ax1.set_title("Phase A: Turing Gray-Scott", fontsize=title_fontsize)
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(imgB, origin='lower', aspect='auto')
ax2.set_title("Phase B: anisotropic diffusion\nof the colorant factor", fontsize=title_fontsize)
ax2.axis('off')

ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(imgC, origin='lower', aspect='auto')
ax3.set_title("Phase C: bistable coloring", fontsize=title_fontsize)
ax3.axis('off')

ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.plot_surface(X, Y, Vn, cmap=cmap_A, linewidth=0, antialiased=False, alpha=0.9)
ax4.set_zlabel("V")
ax4.set_box_aspect((1, 1, ratio_z))

ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.plot_surface(X_hr, Y_hr, colorant, cmap=cmap_B, linewidth=0, antialiased=False, alpha=0.9)
ax5.set_zlabel("Colorant concentration")
ax5.set_box_aspect((1, 1, ratio_z))
y_src, x_src = np.where(sources)
z_src = colorant[y_src, x_src]
for xi, yi, zi in zip(x_src, y_src, z_src):
    ax5.plot([xi], [yi], [zi], marker='o', color='yellow', markersize=2, zorder=999)

ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.plot_surface(X_hr, Y_hr, hill_response, cmap=cmap_C, linewidth=0, antialiased=False)
ax6.set_zlabel("Pigment response")
ax6.set_box_aspect((1, 1, ratio_z))

for ax in (ax4, ax5, ax6):
    ax.view_init(elev=30, azim=-60)
    ax.set_zticks(np.arange(0, 1.1, 0.4))

plt.tight_layout()

for label, (x, y) in [("A", (0.02, 1.03)), ("B", (0.02, 0.40))]:
    fig.text(x, y, label, fontsize=64, fontweight='bold', transform=fig.transFigure)
for j, sub in enumerate(["(i)", "(ii)", "(iii)"]):
    fig.text(0.18 + j * 0.31, 1.01, sub, fontsize=32, fontweight='bold', transform=fig.transFigure)
    fig.text(0.18 + j * 0.31, 0.42, sub, fontsize=32, fontweight='bold', transform=fig.transFigure)

#%%
# =============================================================================
# EXPORTS  (filenames suffixed _extended so as not to overwrite the exports
# of fashion_model.py, which remains the 7-vein reference)
# =============================================================================
output_dir = Path(__file__).resolve().parent

np.save(output_dir / "pattern_phaseA_V_nervured_extended.npy",     V)
np.save(output_dir / "pattern_phaseB_colorant_extended.npy",      colorant)
np.save(output_dir / "pattern_phaseC_hill_response_extended.npy", hill_response)

np.savetxt(output_dir / "pattern_phaseA_V_nervured_extended.csv", V,            delimiter=",")
np.savetxt(output_dir / "pattern_phaseB_colorant_extended.csv",   colorant,     delimiter=",")
np.savetxt(output_dir / "pattern_phaseC_hill_response_extended.csv", hill_response, delimiter=",")

np.savetxt(output_dir / "nervure_positions_extended.csv", nervure_positions_hr,
           delimiter=",", header="x_px", comments="", fmt="%d")

init_points_arr = np.array(init_points)
np.savetxt(output_dir / "seed_points_extended.csv", init_points_arr,
           delimiter=",", header="y_px,x_px", comments="", fmt="%d")

print(f"Matrices saved to: {output_dir}")

plt.show()
