"""
fashion_model_animation.py
======================
Animation of the three-phase patterning model of the snake's head
fritillary (Fritillaria meleagris). Uses the same parameters and physics
as fashion_model.py (highres_factor=3, F=0.0382, n_steps=60e3, pad+wrap
Laplacian, optional Numba).

Movie structure
-----------------
  Phase A     : animation of the Gray-Scott simulation (n_frames_A snapshots)
  Transition  : animated 3D cutting plane showing threshold activation
  Phase B     : anisotropic diffusion of the colorant factor
  Phase C     : interpolation towards the Hill response (bistability)

Requires the ffmpeg binary on PATH (used by matplotlib's FFMpegWriter to
encode the .mp4).
"""

import sys
import io

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pathlib import Path

# Numba is optional: speeds up the Gray-Scott loop via JIT compilation.
try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*args, **kwargs):           # no-op decorator
        return lambda f: f

#%%
# =============================================================================
# MAIN PARAMETERS  (taken from fashion_model.py)
# =============================================================================

# --- Spatial domain (reference scale, rescaled by highres_factor) ---
height_base, width_base = 50, 50
highres_factor = 3        # 1 = original 50x50 | 3 = 150x150 (as in fashion_model.py)

# --- Phase A: Gray-Scott ---
n_steps_A  = int(60e3)    # same duration as fashion_model.py
dt_base    = 1.0
dx_base    = 1.0
Du, Dv     = 0.14, 0.06
F,  k      = 0.0382, 0.065
n_frames_A = 60            # number of snapshots captured during phase A

# --- Transition A->B ---
transition_frames = 40     # total transition frames (split into 2 stages)

# --- Phase B: anisotropic diffusion ---
Dx_color  = 1.0
Dy_color  = 0.1
dt_color  = 0.05
n_iter_B  = 350 * highres_factor   # rescaled as in fashion_model.py
frame_every_B = max(1, n_iter_B // 60)   # ~60 frames for phase B

# --- Phase C ---
n_steps_C_interp = 60      # interpolation frames towards the Hill response
n_Hill, K_Hill   = 3.0, 0.05

# --- Veins ---
n_nervures          = 7
nervure_strength    = 0.4
nervure_width_px    = 2
nervure_color_width = 1
nervures_alpha      = 0.25

# --- Palette ---
yellow    = np.array([1.0,  1.0,  0.0])
black     = np.array([0.0,  0.0,  0.0])

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

from matplotlib.colors import LinearSegmentedColormap
CMAP_A = LinearSegmentedColormap.from_list("phaseA", [PHASE_A_LIGHT, PHASE_A_DARK])
CMAP_B = LinearSegmentedColormap.from_list("phaseB", [PHASE_B_LIGHT, PHASE_B_DARK])
CMAP_C = LinearSegmentedColormap.from_list("phaseC", [PHASE_C_DARK, PHASE_C_LIGHT])

# --- Rendering ---
z_ratio    = 0.15
fps        = 8
dpi        = 150
movie_name = "fashion_model_animation.mp4"

#%%
# =============================================================================
# PHYSICAL SCALING  (same logic as fashion_model.py)
# =============================================================================
height = height_base * highres_factor
width  = width_base  * highres_factor

# dx reduced -> physically finer grid, non-pixelated diffusion
# dt reduced by highres_factor^2 to satisfy the CFL condition (dt <= dx^2/2D_max)
dx = dx_base / highres_factor
dt = dt_base / highres_factor**2

nervure_width_px_scaled    = max(1, nervure_width_px    * highres_factor)
nervure_color_width_scaled = max(1, nervure_color_width * highres_factor)

# Seed patch: preserves the same physical size as at scale 1
seed_radius = max(1, highres_factor - 1)

print(f"Grid: {width}x{height}  |  dx={dx:.4f}  dt={dt:.6f}  n_steps_A={n_steps_A}"
      f"  |  Numba: {'yes' if _NUMBA else 'no'}")

#%%
# =============================================================================
# INITIALIZATION
# =============================================================================
U = np.ones((height, width))
V = np.zeros((height, width))

nervure_positions = np.linspace(0, width - 1, n_nervures, dtype=int)

Du_map = Du * np.ones_like(U)
Dv_map = Dv * np.ones_like(V)
for x_pos in nervure_positions:
    s = max(0, x_pos - nervure_width_px_scaled // 2)
    e = min(width,  x_pos + nervure_width_px_scaled // 2)
    Du_map[:, s:e] *= (1 - nervure_strength)
    Dv_map[:, s:e] *= (1 - nervure_strength)

# Seeds: physically constant patch (same area as a point at scale 1)
init_points = []
for i in range(n_nervures - 1):
    x_center = (nervure_positions[i] + nervure_positions[i + 1]) // 2
    y_center  = height - 2
    for dy in range(-seed_radius, seed_radius + 1):
        for dx_ in range(-seed_radius, seed_radius + 1):
            yy = np.clip(y_center + dy, 0, height - 1)
            xx = np.clip(x_center + dx_, 0, width  - 1)
            U[yy, xx] = 2.0
            V[yy, xx] = 0.75
    init_points.append((y_center, x_center))

#%%
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
inv_dx2 = 1.0 / dx**2

@njit(cache=True)
def gs_step(U, V, Du_map, Dv_map, F, k, dt, inv_dx2):
    """One Gray-Scott step with a 5-point Laplacian (periodic wrap boundaries)."""
    Up = np.empty((U.shape[0] + 2, U.shape[1] + 2))
    Vp = np.empty((V.shape[0] + 2, V.shape[1] + 2))
    Up[1:-1, 1:-1] = U; Up[0, 1:-1] = U[-1, :]; Up[-1, 1:-1] = U[0, :]
    Up[:, 0] = Up[:, -2]; Up[:, -1] = Up[:, 1]
    Vp[1:-1, 1:-1] = V; Vp[0, 1:-1] = V[-1, :]; Vp[-1, 1:-1] = V[0, :]
    Vp[:, 0] = Vp[:, -2]; Vp[:, -1] = Vp[:, 1]

    Lu = (-4*U + Up[2:, 1:-1] + Up[:-2, 1:-1] + Up[1:-1, 2:] + Up[1:-1, :-2]) * inv_dx2
    Lv = (-4*V + Vp[2:, 1:-1] + Vp[:-2, 1:-1] + Vp[1:-1, 2:] + Vp[1:-1, :-2]) * inv_dx2
    UVV = U * V**2
    U = U + dt * (Du_map * Lu - UVV + F * (1 - U))
    V = V + dt * (Dv_map * Lv + UVV - (F + k) * V)

    U[0, :]  = U[1, :];  U[-1, :] = U[-2, :]
    U[:, 0]  = U[:, 1];  U[:, -1] = U[:, -2]
    V[0, :]  = V[1, :];  V[-1, :] = V[-2, :]
    V[:, 0]  = V[:, 1];  V[:, -1] = V[:, -2]
    return U, V

def draw_nervures(img, alpha=1.0, color=black):
    """Overlay the veins on an RGB image with alpha transparency."""
    out = img.copy()
    for xp in nervure_positions:
        s = max(0, int(xp - nervure_color_width_scaled // 2))
        e = min(width, int(xp + nervure_color_width_scaled // 2 + 1))
        out[:, s:e, :] = (1 - alpha) * out[:, s:e, :] + alpha * color
    return out

def draw_seeds(img):
    """Mark the seed points in yellow on an RGB image."""
    out = img.copy()
    for (y, x) in init_points:
        for dy in range(-seed_radius, seed_radius + 1):
            for dx_ in range(-seed_radius, seed_radius + 1):
                yy = np.clip(y + dy, 0, height - 1)
                xx = np.clip(x + dx_, 0, width  - 1)
                out[yy, xx] = yellow
    return out

def progress_bar(step, total, label="", width_bar=40):
    """Display a console progress bar (no external dependency)."""
    pct  = (step + 1) / total
    done = int(width_bar * pct)
    bar  = "█" * done + "░" * (width_bar - done)
    sys.stdout.write(f"\r  [{bar}] {100*pct:5.1f}%  {label} {step+1}/{total}")
    sys.stdout.flush()

# Frame lists: 2D RGB image, 2D scalar data, metadata
frames_img2d  = []
frames_vdata  = []
frames_meta   = []   # tuples (phase_name, auxiliary_value)

#%%
# =============================================================================
# PHASE A: GRAY-SCOTT REACTION-DIFFUSION
# =============================================================================
print("Phase A: Gray-Scott simulation")
# Snapshots regularly spaced over the whole simulation
snapshot_steps = set(np.linspace(0, n_steps_A, n_frames_A, dtype=int))

# Compositing is deferred until after the loop: V's own peak (~0.4-0.5, never
# actually reaches 1) is only known once the run is complete, and normalizing
# each snapshot by ITS OWN max independently would make early, barely-formed
# frames look just as saturated as the mature final one -- defeating the
# point of an animation showing gradual growth. Every Phase A frame is
# instead normalized by the SAME final V.max(), computed once at the end.
phaseA_raw_V   = []
phaseA_has_seeds = []

for step in range(n_steps_A + 1):
    U, V = gs_step(U, V, Du_map, Dv_map, F, k, dt, inv_dx2)

    if step in snapshot_steps:
        phaseA_raw_V.append(V.copy())
        phaseA_has_seeds.append(True)
        frames_meta.append(("Phase A", step))

    progress_bar(step, n_steps_A + 1, "Gray-Scott")

print()

if np.isnan(U).any() or np.isnan(V).any():
    raise ValueError("Numerical instability detected: NaN in U or V. "
                     "Check the parameters (dt too large, or Du/Dv too high).")

# Final phase A frame (without seed points, for the transition)
phaseA_raw_V.append(V.copy())
phaseA_has_seeds.append(False)
frames_meta.append(("Phase A finale", n_steps_A))

# Single normalization constant for the whole Phase A sequence -- V never
# actually reaches 1 (Gray-Scott's V typically peaks around 0.4-0.5 in this
# parameter regime), so np.clip(V, 0, 1) alone was a no-op that left every
# frame under-saturated relative to the intended white->bud-green gradient.
vmax_A = max(phaseA_raw_V[-1].max(), 1e-9)
for V_snap, has_seeds in zip(phaseA_raw_V, phaseA_has_seeds):
    Vn  = np.clip(V_snap / vmax_A, 0, 1)
    img = PHASE_A_LIGHT * (1 - Vn[..., None]) + PHASE_A_DARK * Vn[..., None]
    img = draw_nervures(img)
    if has_seeds:
        img = draw_seeds(img)
    frames_img2d.append(img)
    frames_vdata.append(Vn.copy())

print("Phase A done.")

#%%
# =============================================================================
# TRANSITION A->B: 3D CUTTING PLANE  (black plane + yellow intersection)
# =============================================================================
print("Transition A->B: 3D cutting plane")
V_threshold = 0.9 * V.max()   # colorant activation threshold (raw V units --
                               # correct as-is for `sources = V >= V_threshold`
                               # below, which still compares against raw V)
last_data   = frames_vdata[-1]
# frames_vdata for Phase A frames now holds Vn = V/vmax_A (normalized 0-1,
# see the Phase A block above), not raw V -- so the 3D transition plots
# below, which plot/threshold `data3` (== last_data, a Phase A frame), need
# the SAME threshold expressed in that normalized scale, or the cutting
# plane ends up positioned far too low relative to the now-taller peaks.
V_threshold_norm = V_threshold / vmax_A

# Step 1: black plane arrives progressively (n_transition_surface frames)
n_transition_surface      = transition_frames // 2
n_transition_intersection = transition_frames - n_transition_surface

for t in range(n_transition_surface):
    frames_img2d.append(frames_img2d[-1].copy())
    frames_vdata.append(last_data.copy())
    frames_meta.append(("Transition Surface", t))      # t = index for the 3D animation

for t in range(n_transition_intersection):
    frames_img2d.append(frames_img2d[-1].copy())
    frames_vdata.append(last_data.copy())
    frames_meta.append(("Transition Intersection", t))

#%%
# =============================================================================
# PHASE B: ANISOTROPIC DIFFUSION OF THE COLORANT FACTOR
# =============================================================================
print("Phase B: colorant diffusion")
sources = (V >= V_threshold)
colorant = np.zeros_like(V)
colorant[sources] = 1.0

# Permeability mask: veins block the colorant
permeable_mask = np.ones((height, width), dtype=bool)
for xp in nervure_positions:
    s = max(0, int(xp - nervure_width_px_scaled // 2))
    e = min(width, int(xp + nervure_width_px_scaled // 2 + 1))
    permeable_mask[:, s:e] = False

for it in range(n_iter_B):
    Cx = (np.roll(colorant, -1, axis=1) + np.roll(colorant, 1, axis=1) - 2*colorant) * Dx_color
    Cy = (np.roll(colorant, -1, axis=0) + np.roll(colorant, 1, axis=0) - 2*colorant) * Dy_color
    colorant += dt_color * (Cx + Cy)
    colorant[~permeable_mask] = 0.0   # vein impermeability
    colorant[sources]          = 1.0   # reinjection at the sources
    colorant = np.clip(colorant, 0.0, 1.0)

    if it % frame_every_B == 0:
        imgB = PHASE_B_LIGHT * (1 - colorant[..., None]) + PHASE_B_DARK * colorant[..., None]
        imgB[sources] = yellow
        imgB = draw_nervures(imgB, alpha=nervures_alpha)
        frames_img2d.append(imgB)
        frames_vdata.append(colorant.copy())
        frames_meta.append(("Phase B", it))

    progress_bar(it, n_iter_B, "Phase B")

print()
print("Phase B done.")

#%%
# =============================================================================
# PHASE C: SOFT BISTABILITY (INTERPOLATION TOWARDS THE HILL RESPONSE)
# =============================================================================
print("Phase C: Hill bistable")
last_color = frames_vdata[-1].copy()
hill_final = (last_color**n_Hill) / (last_color**n_Hill + K_Hill**n_Hill)

# Suppress pigment inside the veins
for xp in nervure_positions:
    s = max(0, int(xp - nervure_width_px_scaled // 2))
    e = min(width, int(xp + nervure_width_px_scaled // 2 + 1))
    hill_final[:, s:e] = 0.0

for t in range(n_steps_C_interp):
    frac  = (t + 1) / n_steps_C_interp
    interp = (1 - frac) * last_color + frac * hill_final
    imgC  = PHASE_C_DARK * (1 - interp[..., None]) + PHASE_C_LIGHT * interp[..., None]
    imgC  = draw_nervures(imgC, alpha=nervures_alpha)
    frames_img2d.append(imgC)
    frames_vdata.append(interp.copy())
    frames_meta.append(("Phase C", t))

print(f"Phase C done -- {len(frames_img2d)} total frames")

#%%
# =============================================================================
# VIDEO RENDERING
# =============================================================================
print("Encoding video...")

# Display labels for each phase
nom_affichage = {
    "Phase A"               : "Phase A:\nTuring patterning (Gray-Scott)",
    "Phase A finale"        : "Phase A:\nInhibitor stabilization (green)",
    "Transition Surface"    : "Transition:\nThreshold activation (gray)",
    "Transition Intersection": "Transition:\nDiffusion sources (yellow)",
    "Phase B"               : "Phase B:\nColoring factor diffusion",
    "Phase C"               : "Phase C:\nDelimitation by bistable amplification",
}

fig     = plt.figure(figsize=(10, 5))
ax_img  = fig.add_subplot(1, 2, 1)
ax_3d   = fig.add_subplot(1, 2, 2, projection="3d")
ax_img.axis("off")
ax_3d.set_box_aspect((1, 1, z_ratio))
X, Y = np.meshgrid(np.arange(width), np.arange(height))

im2d  = ax_img.imshow(frames_img2d[0], origin="lower")
texte = ax_img.text(0.02, 0.98, "", transform=ax_img.transAxes,
                    color="white", fontsize=10, va="top",
                    bbox=dict(facecolor="black", alpha=0.4))

writer = FFMpegWriter(fps=fps, bitrate=None)
output_path = Path(__file__).resolve().parent / movie_name

with writer.saving(fig, str(output_path), dpi=dpi):
    for i in range(len(frames_img2d)):
        img   = frames_img2d[i]
        data3 = frames_vdata[i]
        phase = frames_meta[i][0]
        t_aux = frames_meta[i][1]   # auxiliary index for the transitions

        # --- Phase text ---
        texte.set_text(nom_affichage.get(phase, phase))
        im2d.set_array(img)

        # --- 3D part ---
        ax_3d.cla()
        ax_3d.set_box_aspect((1, 1, z_ratio))
        ax_3d.set_zlim(0, 1.05)
        ax_3d.view_init(30, -60)
        ax_3d.set_zticks(np.arange(0, 1.1, 0.4))

        if phase == "Transition Surface":
            # Black plane arriving progressively along Y
            frac    = min((t_aux + 1) / n_transition_surface, 1.0)
            Y_lim   = Y.min() + frac * (Y.max() - Y.min())
            Y_masked = np.where(Y <= Y_lim, Y, np.nan)

            Z_above = np.where(data3 >= V_threshold_norm, data3, np.nan)
            Z_below = np.where(data3 <  V_threshold_norm, data3, np.nan)
            ax_3d.plot_surface(X, Y, Z_above, cmap=CMAP_A, linewidth=0,
                               antialiased=False, alpha=1.0)
            ax_3d.plot_surface(X, Y, Z_below, cmap=CMAP_A, linewidth=0,
                               antialiased=False, alpha=0.4)
            ax_3d.plot_surface(X, Y_masked, np.full_like(data3, V_threshold_norm),
                               color='black', alpha=0.3, linewidth=0)

        elif phase == "Transition Intersection":
            # Fading surface + yellow contour of the intersection
            frac    = (t_aux + 1) / n_transition_intersection
            Z_above = np.where(data3 >= V_threshold_norm, data3, np.nan)
            Z_below = np.where(data3 <  V_threshold_norm, data3, np.nan)
            ax_3d.plot_surface(X, Y, Z_above, cmap=CMAP_A, linewidth=0,
                               antialiased=False, alpha=(1 - frac))
            ax_3d.plot_surface(X, Y, Z_below, cmap=CMAP_A, linewidth=0,
                               antialiased=False, alpha=0.5 * (1 - frac))
            ax_3d.plot_surface(X, Y, np.full_like(data3, V_threshold_norm),
                               color='black', alpha=0.3, linewidth=0)
            ax_3d.contour(X, Y, data3, levels=[V_threshold_norm],
                          colors='yellow', linewidths=2, offset=V_threshold_norm)

        elif phase == "Phase B":
            # Diffusion surface + yellow source points
            ax_3d.plot_surface(X, Y, data3, cmap=CMAP_B, linewidth=0,
                               antialiased=False, alpha=1.0)
            y_src, x_src = np.where(sources)
            z_src = data3[y_src, x_src]
            for xi, yi, zi in zip(x_src, y_src, z_src):
                ax_3d.plot([xi], [yi], [zi], marker='o', color='yellow',
                           markersize=3, zorder=999)

        elif phase == "Phase C":
            # Hill-response surface
            ax_3d.plot_surface(X, Y, data3, cmap=CMAP_C, linewidth=0,
                               antialiased=False, alpha=1.0)

        else:
            # Phases A, A final: simple surface
            ax_3d.plot_surface(X, Y, data3, cmap=CMAP_A, linewidth=0,
                               antialiased=False, alpha=1.0)

        writer.grab_frame()
        progress_bar(i, len(frames_img2d), "Encoding")

print(f"\nVideo saved: {output_path}")
