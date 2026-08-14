"""
turing_pattern_vein_free_extended.py
======================================
"Large domain" variant of turing_pattern_vein_free.py: the same pure
Turing pattern with no vein effect on diffusion, but with n_nervures = 15
instead of 7 (same pixel resolution, larger physical domain -- same
width_base/height_base/highres_factor as fashion_model_extended.py, to
stay comparable to the "with veins" group at this scale). Does not modify
turing_pattern_vein_free.py.

Differences from turing_pattern_vein_free.py
-----------------------------------------------
  - width_base = height_base = 115 (instead of 50x50), n_steps = 184,000
    (80e3 * height_base/50): same scaling as the with-veins version (see
    fashion_model_extended.py).
  - Phase A: F=0.03875/k=0.065 work as-is, with no retuning -- there is
    no propagation blockage here (the "holes" and compartment merges
    observed when scaling up the with-veins version come from
    competition between inter-vein compartments, a mechanism absent here
    since nervure_strength=0, i.e. no physical compartmentalization; a
    single origin would therefore be largely sufficient to fill the
    whole domain).
  - seed_rows = 2 (top+bottom), **identical to fashion_model_extended.py,
    for methodological rather than physical reasons**: a single row
    would be enough here (see previous point), but the seeding protocol
    must stay rigorously the same in both scripts so that the with/
    without-veins comparison isolates the effect of the veins and
    nothing else -- otherwise an observed difference between the two
    could come from either condition, not necessarily from the veins.
  - Phase B: the per-compartment source threshold (as in the with-veins
    version) is kept for methodological consistency, but has no
    corrective effect here -- there is no inter-compartment competition
    to correct, just a visual/statistical comparison grid.
  - Phase B: **n_iter reduced to 230*highres_factor (instead of
    350*highres_factor)** -- this one did need to change. With no veins
    to contain lateral diffusion, the Turing field self-organizes into a
    dense hexagonal lattice (Gray-Scott's natural wavelength, not the
    grid imposed by the veins) -- the spots are closer together than at
    the reference scale. With the reference value (350), lateral
    diffusion over-spreads each source and merges the whole domain into
    a single blob riddled with holes instead of distinct spots (visible
    only at HD; at half resolution it still looks clean -- diagnosed by
    comparing the raw, clean V field to the broken final phase C render,
    on the same run).

High-resolution physics: identical to fashion_model_extended.py.

Measured runtime: on the order of several hours (this is NOT the fast
classroom script -- see fashion_model.py and turing_pattern_vein_free.py
for that).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%%
# =============================================================================
# PARAMETERS
# =============================================================================
height_base = 115
width_base  = 115
n_nervures  = 15

highres_factor = 3   # must match fashion_model_extended.py

n_steps = int(184e3)  # 80e3 * height_base/50 (rescaled, see docstring)
dt_base = 1.0
dx_base = 1.0

Du = 0.14
Dv = 0.06
F  = 0.03875
k  = 0.065

nervure_strength    = 0.0   # key: no vein effect on Gray-Scott
nervure_width_px    = 1     # matched to fashion_model_extended.py (decorative grid here)
nervure_color_width = 1

# seed_rows = 2 (top+bottom), IDENTICAL to fashion_model_extended.py -- even
# though a single row would be enough here (no compartments to cross, see
# docstring), the seeding protocol must stay rigorously identical between
# the two scripts so that the with/without-veins comparison isolates the
# effect of the veins and nothing else.
seed_rows = 2

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
# PHYSICAL SCALING  (identical to fashion_model_extended.py)
# =============================================================================
height = height_base * highres_factor
width  = width_base  * highres_factor

dx = dx_base / highres_factor
dt = dt_base / highres_factor**2

PIXEL_SIZE_MM = 0.05 / highres_factor

nervure_width_px_scaled    = max(1, nervure_width_px    * highres_factor)
nervure_color_width_scaled = max(1, nervure_color_width * highres_factor)
seed_radius                = max(1, highres_factor - 1)

print(f"Grid: {width}x{height}  |  dx={dx:.4f}  dt={dt:.6f}  n_steps={n_steps}")
print(f"Pixel size: {PIXEL_SIZE_MM*1000:.1f} um  |  nervure_strength={nervure_strength}")

#%%
# =============================================================================
# PHASE A: VEIN-FREE GRAY-SCOTT
# =============================================================================
U = np.ones((height, width))
V = np.zeros((height, width))

nervure_positions = np.linspace(0, width - 1, n_nervures, dtype=int)

Du_map = Du * np.ones_like(U)
Dv_map = Dv * np.ones_like(V)
for x_pos in nervure_positions:
    s = max(0, x_pos - nervure_width_px_scaled // 2)
    e = min(width, x_pos + nervure_width_px_scaled // 2)
    Du_map[:, s:e] *= (1 - nervure_strength)   # = x1.0, no effect
    Dv_map[:, s:e] *= (1 - nervure_strength)

init_points = []
y_centers = (height - 2, 1) if seed_rows == 2 else [height - 2]
for y_center in y_centers:
    for i in range(n_nervures - 1):
        x_center = (nervure_positions[i] + nervure_positions[i + 1]) // 2
        for dy in range(-seed_radius, seed_radius + 1):
            for dx_ in range(-seed_radius, seed_radius + 1):
                yy = np.clip(y_center + dy, 0, height - 1)
                xx = np.clip(x_center + dx_, 0, width  - 1)
                U[yy, xx] = 1.0
                V[yy, xx] = 0.5
        init_points.append((y_center, x_center))

inv_dx2 = 1.0 / dx**2

def laplacian(Z):
    """5-point Laplacian, periodic boundaries (pad+slice)."""
    Zp = np.pad(Z, 1, mode='wrap')
    return (-4*Z + Zp[2:,1:-1] + Zp[:-2,1:-1] + Zp[1:-1,2:] + Zp[1:-1,:-2]) * inv_dx2

print("Phase A: Gray-Scott simulation (vein-free)")
report_every  = max(1, n_steps // 50)
bar_width_con = 40

for step in range(n_steps):
    Lu  = laplacian(U)
    Lv  = laplacian(V)
    UVV = U * V**2
    U  += dt * (Du_map * Lu - UVV + F * (1 - U))
    V  += dt * (Dv_map * Lv + UVV - (F + k) * V)
    for Z in (U, V):
        Z[0,:]  = Z[1,:];  Z[-1,:] = Z[-2,:]
        Z[:,0]  = Z[:,1];  Z[:,-1] = Z[:,-2]

    if step % report_every == 0 or step == n_steps - 1:
        pct  = (step + 1) / n_steps
        done = int(bar_width_con * pct)
        sys.stdout.write(f"\r  [{'█'*done}{'░'*(bar_width_con-done)}] {100*pct:5.1f}%  step {step+1}/{n_steps}")
        sys.stdout.flush()

print("\nPhase A done.")

if np.isnan(U).any() or np.isnan(V).any():
    raise ValueError("Numerical instability: NaN in U or V.")

# Divide by V.max() first: Gray-Scott's V never actually reaches 1 in this
# regime (typically peaks ~0.4-0.5), so np.clip(V, 0, 1) alone was a no-op
# that left the pattern permanently under-saturated.
Vn   = np.clip(V / max(V.max(), 1e-9), 0, 1)
imgA = PHASE_A_LIGHT * (1 - Vn[..., None]) + PHASE_A_DARK * Vn[..., None]
for xp in nervure_positions:
    s = max(0, int(xp - nervure_color_width_scaled // 2))
    e = min(width, int(xp + nervure_color_width_scaled // 2 + 1))
    imgA[:, s:e, :] = 0.5 * imgA[:, s:e, :] + 0.5 * black
imgA_with_anchors = imgA.copy()
for (y, x) in init_points:
    for dy in range(-seed_radius, seed_radius + 1):
        for dx_ in range(-seed_radius, seed_radius + 1):
            yy = np.clip(y + dy, 0, height - 1)
            xx = np.clip(x + dx_, 0, width  - 1)
            imgA_with_anchors[yy, xx] = yellow

#%%
# =============================================================================
# PHASES B & C: same HR grid, same logic as fashion_model_extended.py
# =============================================================================
height_hr              = height
width_hr                = width
nervure_positions_hr    = nervure_positions
nervure_width_px_hr     = nervure_width_px_scaled
nervure_color_width_hr  = nervure_color_width_scaled

# Per-compartment source threshold (see docstring) -- has no corrective
# effect here (nervure_strength=0), kept for consistency with the
# with-veins version.
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
    block  = V[:, cs:ce]
    if tissue.any():
        local_max = block[tissue].max()
        if local_max > 0:
            sources[:, cs:ce] = (block >= 0.9 * local_max) & tissue

colorant = np.zeros((height_hr, width_hr), dtype=float)
colorant[sources] = 1.0

Dx = 1.0
Dy = 0.15

# In the vein-free version, the permeability mask has no effect either
# (nervure_strength=0, so veins do not block colorant diffusion either)
permeable_mask = np.ones((height_hr, width_hr), dtype=bool)

# n_iter reduced relative to turing_pattern_vein_free.py (350*highres_factor):
# with no veins to contain lateral diffusion (unlike the with-veins
# version), the free Turing field self-organizes into a dense hexagonal
# lattice (Gray-Scott's natural wavelength, not the grid imposed by the
# veins) -- the spots are therefore much closer together than at the
# reference scale. With n_iter=350*highres_factor (the reference value),
# lateral diffusion (Dx=1.0) over-spreads each source and merges the
# whole domain into a single hole-riddled blob instead of distinct spots.
# A sweep on the final V field (150 to 1050 steps): peak quality around
# 700-750 iterations, clear degradation beyond ~950.
# 230*highres_factor=690 stays within the safety margin.
n_iter   = 230 * highres_factor
dt_color = 0.05

print("Phase B: colorant factor diffusion")
report_every_B = max(1, n_iter // 50)

for it in range(n_iter):
    Cx = (np.roll(colorant, -1, axis=1) + np.roll(colorant, 1, axis=1) - 2*colorant) * Dx
    Cy = (np.roll(colorant, -1, axis=0) + np.roll(colorant, 1, axis=0) - 2*colorant) * Dy
    colorant += dt_color * (Cx + Cy)
    colorant[~permeable_mask] = 0.0
    colorant[sources]          = 1.0
    colorant = np.clip(colorant, 0.0, 1.0)

    if it % report_every_B == 0 or it == n_iter - 1:
        pct  = (it + 1) / n_iter
        done = int(bar_width_con * pct)
        sys.stdout.write(f"\r  [{'█'*done}{'░'*(bar_width_con-done)}] {100*pct:5.1f}%  it {it+1}/{n_iter}")
        sys.stdout.flush()

print("\nPhase B done.")

imgB = PHASE_B_LIGHT[None,None,:] * (1 - colorant[...,None]) + PHASE_B_DARK[None,None,:] * colorant[...,None]
imgB[sources] = yellow
alpha = nervures_alpha
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_color_width_hr // 2))
    e = min(width_hr, int(xp + nervure_color_width_hr // 2 + 1))
    imgB[:, s:e, :] = (1 - alpha) * imgB[:, s:e, :] + alpha * gray

n_Hill = 3.0
K_Hill = 0.05
hill_response = (colorant**n_Hill) / (colorant**n_Hill + K_Hill**n_Hill)

imgC = PHASE_C_DARK[None,None,:] * (1 - hill_response[...,None]) + PHASE_C_LIGHT[None,None,:] * hill_response[...,None]
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_color_width_hr // 2))
    e = min(width_hr, int(xp + nervure_color_width_hr // 2 + 1))
    imgC[:, s:e, :] = (1 - alpha) * imgC[:, s:e, :] + alpha * black

#%%
# =============================================================================
# DISPLAY
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
ax1.set_title("Phase A: Turing Gray-Scott (vein-free)", fontsize=title_fontsize)
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

for lbl, (x, y) in [("A", (0.02, 1.03)), ("B", (0.02, 0.40))]:
    fig.text(x, y, lbl, fontsize=64, fontweight='bold', transform=fig.transFigure)
for j, sub in enumerate(["(i)", "(ii)", "(iii)"]):
    fig.text(0.18 + j*0.31, 1.01, sub, fontsize=32, fontweight='bold', transform=fig.transFigure)
    fig.text(0.18 + j*0.31, 0.42, sub, fontsize=32, fontweight='bold', transform=fig.transFigure)

#%%
# =============================================================================
# EXPORTS  (filenames suffixed _extended so as not to overwrite the exports
# of turing_pattern_vein_free.py, which remains the 7-vein reference)
# =============================================================================
output_dir = Path(__file__).resolve().parent

np.save(output_dir / "pattern_phaseA_V_extended.npy",      V)
np.savetxt(output_dir / "pattern_phaseA_V_extended.csv",   V,             delimiter=",")

np.save(output_dir / "pattern_phaseB_colorant_nervureless_extended.npy",    colorant)
np.savetxt(output_dir / "pattern_phaseB_colorant_nervureless_extended.csv", colorant, delimiter=",")

np.save(output_dir / "pattern_phaseC_hill_response_nervureless_extended.npy", hill_response)
np.savetxt(output_dir / "pattern_phaseC_hill_response_nervureless_extended.csv", hill_response, delimiter=",")
print(f"Matrices saved to: {output_dir}")

plt.show()
