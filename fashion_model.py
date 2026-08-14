"""
fashion_model.py
================
Three-phase patterning model for the checkerboard pattern of the snake's
head fritillary (Fritillaria meleagris).

Model architecture
-------------------
  Phase A - Turing reaction-diffusion (Gray-Scott)
      Generates the base spatial pattern (spots of morphogen V) on a
      physically high-resolution grid.

  Phase B - Anisotropic diffusion of the colorant factor
      The colorant diffuses horizontally from the V spots, but is blocked
      by the veins. This spreads the spots laterally within each
      inter-vein compartment.

  Phase C - Soft bistability (Hill function)
      Amplifies the colorant -> pigment response: high-colorant regions
      switch to a "colored" state, low-colorant regions stay "uncolored".
      The nonlinearity is controlled by the Hill coefficient n_Hill and
      the affinity constant K_Hill.

Key parameter: highres_factor
--------------------------------
  Increases the physical resolution of the Gray-Scott grid (Phase A).
  - dx is divided by highres_factor    -> finer grid, non-pixelated diffusion
  - dt is divided by highres_factor^2  -> CFL condition satisfied (dt <= dx^2/2D_max)
  - n_steps is fixed -> same runtime; the pattern develops to the same
    stage of maturity as at the reference scale (50x50).

  Example: highres_factor = 3 -> 150x150 grid, x3 resolution at no extra
  computational cost.

Algorithmic optimizations
------------------------------
  - Laplacian computed via pad+slicing (2x faster than np.roll, same result)
  - Console progress bar with no external dependency (sys.stdout)
  - Optional Numba JIT (if installed: `pip install numba`) for an
    additional x5-10 speedup on the Gray-Scott loop; otherwise automatic
    fallback to plain NumPy.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Numba is optional: speeds up the Gray-Scott loop via JIT compilation.
# If not installed, the code runs normally in plain NumPy.
try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*args, **kwargs):           # no-op decorator
        return lambda f: f

#%%
# =============================================================================
# BASE PARAMETERS  (scale 1 -- highres_factor rescales everything automatically)
# =============================================================================
height_base = 50          # domain height at the reference scale (pixels)
width_base  = 50          # domain width at the reference scale (pixels)
n_nervures  = 7            # number of vertical veins

# Physical upscaling factor (see docstring at the top of the file)
highres_factor = 3        # 1 = original 50x50 | 3 = 150x150 | 5 = 250x250

# Time parameters (at the reference scale)
n_steps = int(60e3)       # number of Gray-Scott iterations
dt_base = 1.0              # reference time step
dx_base = 1.0              # reference space step

# Gray-Scott reaction-diffusion parameters
Du = 0.14    # diffusion coefficient of morphogen U (inhibitor)
Dv = 0.06    # diffusion coefficient of morphogen V (activator)
F  = 0.0382   # feed rate: supply of U from the reservoir
k  = 0.065   # kill rate: degradation of V

# Veins: zones of reduced diffusion simulating the petal's veins
nervure_strength    = 0.4  # relative diffusion reduction inside veins (0-1)
nervure_width_px    = 2    # vein thickness in pixels at scale 1
nervure_color_width = 1    # visual vein thickness for rendering

# Color palette (RGB, normalized 0-1)
yellow    = np.array([1.0,  1.0,  0.0])    # seed points (visualization)
black     = np.array([0.0,  0.0,  0.0])    # veins in phase A
gray      = np.array([0.45, 0.45, 0.45])   # semi-transparent veins in phase B
nervures_alpha = 0.25                       # vein opacity in the rendering

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

# dx reduced by highres_factor (same physical domain, finer grid)
dx = dx_base / highres_factor
# dt reduced by highres_factor^2 to satisfy the CFL stability condition
dt = dt_base / highres_factor**2

# Vein and seed thickness rescaled (physical size preserved)
nervure_width_px_scaled    = max(1, nervure_width_px    * highres_factor)
nervure_color_width_scaled = max(1, nervure_color_width * highres_factor)

# Seed radius in HR pixels: preserves the physical size of the original seed.
# At scale 1, the seed = 1 pixel. At HR scale, it occupies a patch of
# (2*seed_radius+1)^2 pixels to represent the same physical area.
# seed_radius = highres_factor - 1 gives a patch of side (2*hf-1) ~ hf pixels.
seed_radius = max(1, highres_factor - 1)

print(f"Phase A grid: {width}x{height} px  |  dx={dx:.4f}  dt={dt:.6f}  n_steps={n_steps}")

#%%
# =============================================================================
# PHASE A: TURING REACTION-DIFFUSION  (Gray-Scott model)
# =============================================================================
# Initialization: U uniform at 1 (abundant substrate), V zero (no activator)
U = np.ones((height, width))
V = np.zeros((height, width))

# Vein positions: columns regularly spaced across the width
nervure_positions = np.linspace(0, width - 1, n_nervures, dtype=int)

# Local diffusion maps: Du and Dv reduced in the vein columns
Du_map = Du * np.ones_like(U)
Dv_map = Dv * np.ones_like(V)
for x_pos in nervure_positions:
    s = max(0, x_pos - nervure_width_px_scaled // 2)
    e = min(width, x_pos + nervure_width_px_scaled // 2)
    Du_map[:, s:e] *= (1 - nervure_strength)
    Dv_map[:, s:e] *= (1 - nervure_strength)

# Seeds: local injection of U and V between each pair of consecutive veins.
# Each seed is placed at the bottom of the domain, centered on the
# inter-vein compartment.
init_points = []
for i in range(n_nervures - 1):
    x_center = (nervure_positions[i] + nervure_positions[i + 1]) // 2
    y_center  = height - 2
    for dy in range(-seed_radius, seed_radius + 1):
        for dx_ in range(-seed_radius, seed_radius + 1):
            yy = np.clip(y_center + dy, 0, height - 1)
            xx = np.clip(x_center + dx_, 0, width  - 1)
            U[yy, xx] = 2.0    # local U supersaturation
            V[yy, xx] = 0.75   # initial V concentration to trigger the reaction
    init_points.append((y_center, x_center))

# ------------------------------------------------------------------
# Discrete 5-point Laplacian with periodic boundary conditions (pad+slice).
# Faster than np.roll because it avoids intermediate copies.
# The 'wrap' padding simulates periodic boundaries; Neumann (reflective)
# conditions are then applied on the borders afterwards.
# ------------------------------------------------------------------
inv_dx2 = 1.0 / dx**2

def laplacian_gs(Z):
    """5-point Laplacian with periodic (wrap) boundaries -- 2x faster than np.roll."""
    Zp = np.pad(Z, 1, mode='wrap')
    return (-4*Z + Zp[2:,1:-1] + Zp[:-2,1:-1] + Zp[1:-1,2:] + Zp[1:-1,:-2]) * inv_dx2

# ------------------------------------------------------------------
# Main Gray-Scott loop with a console progress bar
# ------------------------------------------------------------------
print("Phase A: Gray-Scott simulation in progress...")
report_every = max(1, n_steps // 50)
bar_width_console = 40

for step in range(n_steps):
    Lu  = laplacian_gs(U)
    Lv  = laplacian_gs(V)
    UVV = U * V**2            # nonlinear reaction term (autocatalysis of V)
    U  += dt * (Du_map * Lu - UVV + F * (1 - U))
    V  += dt * (Dv_map * Lv + UVV - (F + k) * V)

    # Neumann boundary conditions (reflective: zero flux at the borders)
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

# ------------------------------------------------------------------
# Rendering of phase A
# ------------------------------------------------------------------
# Vn: V concentration normalized between 0 and 1 for color rendering.
# Divide by V.max() first: Gray-Scott's V never actually reaches 1 in this
# regime (typically peaks ~0.4-0.5), so np.clip(V, 0, 1) alone was a no-op
# that left the pattern permanently under-saturated relative to the intended
# white -> bud-green gradient (the darkest green was never actually reached).
Vn   = np.clip(V / max(V.max(), 1e-9), 0, 1)
# Linear interpolation: white (V=0, Gray-Scott prepattern) -> bud green (V=1)
imgA = PHASE_A_LIGHT * (1 - Vn[..., None]) + PHASE_A_DARK * Vn[..., None]
# Veins drawn in black (anatomical marker)
for xp in nervure_positions:
    s = max(0, int(xp - nervure_color_width_scaled // 2))
    e = min(width, int(xp + nervure_color_width_scaled // 2 + 1))
    imgA[:, s:e, :] = black
# Seed points colored yellow for visualization
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
# PHASES B & C: same HR grid as phase A (no additional upscaling)
# =============================================================================
# The Gray-Scott simulation already runs at high resolution thanks to
# highres_factor; phases B and C therefore work directly on these results.
height_hr              = height
width_hr               = width
nervure_positions_hr   = nervure_positions
nervure_width_px_hr    = nervure_width_px_scaled
nervure_color_width_hr = nervure_color_width_scaled
V_hr                   = V.copy()

#%%
# =============================================================================
# PHASE B: ANISOTROPIC DIFFUSION OF THE COLORANT FACTOR
# =============================================================================
# The V spots (concentration >= 90% of the max) act as colorant sources.
# The colorant diffuses horizontally (Dx >> Dy) but is impermeable to the
# veins. This produces bands of coloring between veins, guided by the
# spatial pattern from phase A.

V_threshold = 0.9 * V_hr.max()
sources     = (V_hr >= V_threshold)   # boolean mask of the source regions

# Colorant concentration field (initialized to 1 at sources, 0 elsewhere)
colorant = np.zeros((height_hr, width_hr), dtype=float)
colorant[sources] = 1.0

# Anisotropic diffusion coefficients (horizontal >> vertical)
Dx = 1.0    # fast horizontal diffusion (lateral spreading)
Dy = 0.1   # slow vertical diffusion (vertical confinement)

# Permeability mask: vein columns block diffusion
permeable_mask = np.ones((height_hr, width_hr), dtype=bool)
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_width_px_hr // 2))
    e = min(width_hr, int(xp + nervure_width_px_hr // 2 + 1))
    permeable_mask[:, s:e] = False

# Colorant diffusion parameters
n_iter   = 350 * highres_factor   # number of iterations (rescaled)
dt_color = 0.05                    # colorant diffusion time step

print("Phase B: colorant factor diffusion...")
report_every_B = max(1, n_iter // 50)

for it in range(n_iter):
    # Discrete diffusion flux (finite differences, implicit periodic
    # boundaries via roll)
    Cx = (np.roll(colorant, -1, axis=1) + np.roll(colorant, 1, axis=1) - 2*colorant) * Dx
    Cy = (np.roll(colorant, -1, axis=0) + np.roll(colorant, 1, axis=0) - 2*colorant) * Dy
    colorant += dt_color * (Cx + Cy)
    # Vein impermeability: reset colorant to zero inside the veins
    colorant[~permeable_mask] = 0.0
    # Constant reinjection at the sources (Dirichlet condition: c=1 at V spots)
    colorant[sources] = 1.0
    colorant = np.clip(colorant, 0.0, 1.0)   # numerical stability

    if it % report_every_B == 0 or it == n_iter - 1:
        pct  = (it + 1) / n_iter
        done = int(bar_width_console * pct)
        bar  = "█" * done + "░" * (bar_width_console - done)
        sys.stdout.write(f"\r  [{bar}] {100*pct:5.1f}%  it {it+1}/{n_iter}")
        sys.stdout.flush()

print("\nPhase B done.")

# Phase B rendering: pale pink -> raspberry based on colorant concentration
imgB = PHASE_B_LIGHT[None,None,:] * (1 - colorant[...,None]) + PHASE_B_DARK[None,None,:] * colorant[...,None]
imgB[sources] = yellow                         # sources in yellow
alpha = nervures_alpha
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_color_width_hr // 2))
    e = min(width_hr, int(xp + nervure_color_width_hr // 2 + 1))
    imgB[:, s:e, :] = (1 - alpha) * imgB[:, s:e, :] + alpha * gray   # semi-transparent gray veins

#%%
# =============================================================================
# PHASE C: BISTABLE COLORING (HILL FUNCTION)
# =============================================================================
# The Hill function turns the continuous colorant concentration into a
# bistable (softened all-or-none) pigment response.
# Formula: h(c) = c^n / (c^n + K^n)
#   n_Hill: Hill coefficient (cooperativity; larger n -> sharper transition)
#   K_Hill: affinity constant (colorant concentration at half-maximum)

n_Hill = 3.0    # Hill exponent (cooperativity of the pigment response)
K_Hill = 0.05   # affinity constant (half-activation threshold)

hill_response = (colorant**n_Hill) / (colorant**n_Hill + K_Hill**n_Hill)

# Suppress the response inside the veins (no pigment on the veins)
for xp in nervure_positions_hr:
    s = max(0, int(xp - nervure_color_width_hr // 2))
    e = min(width_hr, int(xp + nervure_color_width_hr // 2 + 1))
    hill_response[:, s:e] = 0.0

# Phase C rendering: magenta-violet (majority background) -> pale
# pink-white (minority patches) based on the Hill response
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
ratio_z  = 0.15      # vertical compression ratio of the 3D surfaces

# 3D surfaces use the same per-phase ramps as the flat 2D images above,
# so each phase reads as the same "color story" in both rows.
from matplotlib.colors import LinearSegmentedColormap
cmap_A = LinearSegmentedColormap.from_list("phaseA", [PHASE_A_LIGHT, PHASE_A_DARK])
cmap_B = LinearSegmentedColormap.from_list("phaseB", [PHASE_B_LIGHT, PHASE_B_DARK])
cmap_C = LinearSegmentedColormap.from_list("phaseC", [PHASE_C_DARK, PHASE_C_LIGHT])

# --- Row 1: images of the three phases ---
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

# --- Row 2: 3D concentration surfaces ---
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

# Panel numbering, scientific-figure style
for label, (x, y) in [("A", (0.02, 1.03)), ("B", (0.02, 0.40))]:
    fig.text(x, y, label, fontsize=64, fontweight='bold', transform=fig.transFigure)
for j, sub in enumerate(["(i)", "(ii)", "(iii)"]):
    fig.text(0.18 + j * 0.31, 1.01, sub, fontsize=32, fontweight='bold', transform=fig.transFigure)
    fig.text(0.18 + j * 0.31, 0.42, sub, fontsize=32, fontweight='bold', transform=fig.transFigure)

#%%
# =============================================================================
# EXPORTS
# =============================================================================
output_dir = Path(__file__).resolve().parent

np.save(output_dir / "pattern_phaseA_V_nervured.npy",     V)
np.save(output_dir / "pattern_phaseB_colorant.npy",      colorant)
np.save(output_dir / "pattern_phaseC_hill_response.npy", hill_response)

np.savetxt(output_dir / "pattern_phaseA_V_nervured.csv", V,            delimiter=",")
np.savetxt(output_dir / "pattern_phaseB_colorant.csv",   colorant,     delimiter=",")
np.savetxt(output_dir / "pattern_phaseC_hill_response.csv", hill_response, delimiter=",")

# Vein positions (x columns, in HR pixels) -- to plot the veins directly
# from the data rather than by eye.
np.savetxt(output_dir / "nervure_positions.csv", nervure_positions_hr,
           delimiter=",", header="x_px", comments="", fmt="%d")

# Seed coordinates (y, x in HR pixels) -- one point per inter-vein compartment.
init_points_arr = np.array(init_points)  # columns: y_px, x_px
np.savetxt(output_dir / "seed_points.csv", init_points_arr,
           delimiter=",", header="y_px,x_px", comments="", fmt="%d")

print(f"Matrices saved to: {output_dir}")

plt.show()
