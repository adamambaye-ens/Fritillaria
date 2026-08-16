[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21944388.svg)](https://doi.org/10.5281/zenodo.21944388)

# FASHION: fritillary checkerboard patterning model

Simulation code accompanying the manuscript *"Tricking Turing: How to
Build a Geometric Checkerboard on Living Tissue"* (J. Primel, P.
Lefebvre, A. Mbaye -- Master de Biologie, ENS de Lyon), currently in
preparation. The model is described there as **FASHION**
(Feedback-driven Anisotropic Self-organized Heterogeneity In Oriented
Networks): a three-phase biophysical model for the violet/white
checkerboard pattern of the snake's head fritillary (*Fritillaria
meleagris*).

This repository contains three self-contained, runnable scripts intended
for direct classroom reuse (fast, small domain, 7 veins), plus two
"large-domain" variants (15 veins) used to generate the quantitative
statistics reported in the paper. Every script runs on its own, from a
fresh Python environment, with no other files or prior steps required.

## Contents

### The three fast scripts (7 veins, minutes to run)

- **`turing_pattern_vein_free.py`** -- classical Gray-Scott Turing-pattern
  simulator (vein-free). Shows what the reaction-diffusion pattern looks
  like on a free, uncompartmentalized domain -- the control case against
  which the vein-driven model is compared.

- **`fashion_model.py`** -- the full three-phase FASHION model: Turing
  reaction-diffusion (phase A) constrained by veins, anisotropic
  colorant diffusion (phase B), and a bistable Hill-function pigment
  response (phase C). Produces the static 3x2 summary figure (2D pattern
  + 3D concentration surface for each phase).

- **`fashion_model_animation.py`** -- animates the same three-phase
  process frame by frame (Gray-Scott simulation, a 3D "cutting-plane"
  transition showing the activation threshold, colorant diffusion, and
  the Hill bistability interpolation), and encodes it as an .mp4.

### Large-domain variants (15 veins, tens of minutes to run)

- **`turing_pattern_vein_free_extended.py`** -- same vein-free control as
  `turing_pattern_vein_free.py`, on the larger 15-vein domain.

- **`fashion_model_extended.py`** -- same FASHION model as
  `fashion_model.py`, on the larger 15-vein domain used for the paper's
  quantitative Square Index / Checkerboard Score statistics. Its
  docstring documents, in detail, every parameter that had to be
  re-tuned to scale the model up from 7 to 15 veins (seeding,
  per-compartment source threshold, vein strength, colorant anisotropy)
  and why -- worth reading even if you only run the fast version.

These two are **not** meant for live classroom use -- see measured
runtimes below. They're included for full transparency and
reproducibility of the paper's quantitative results, not for teaching.

## Dependencies

Core requirement: **NumPy + Matplotlib only** (pure NumPy, no SciPy
dependency in any of these five scripts).

- `numpy`
- `matplotlib` (3D surface plots use `mpl_toolkits.mplot3d`, bundled
  with Matplotlib)
- `numba` (**optional**) -- relevant only to `fashion_model.py`,
  `fashion_model_extended.py`, and `fashion_model_animation.py`, which
  try to import it and fall back to plain NumPy if it isn't installed.
  Of these three, only `fashion_model_animation.py` actually applies
  `@njit` to accelerate its Gray-Scott loop; in the other two the import
  is present but unused, so installing Numba has no effect on them.
  `turing_pattern_vein_free.py` and `turing_pattern_vein_free_extended.py`
  don't reference Numba at all -- plain NumPy only.
- **`ffmpeg`** (system binary, not a Python package) -- required only by
  `fashion_model_animation.py`, via Matplotlib's `FFMpegWriter`. Must be
  on `PATH`. Not needed by the other four scripts.

Install with:

```bash
pip install -r requirements.txt
```

## Running

Each script is standalone:

```bash
python turing_pattern_vein_free.py
python fashion_model.py
python fashion_model_animation.py
python turing_pattern_vein_free_extended.py   # ~30 min
python fashion_model_extended.py              # ~27 min
```

All five generate their own initial conditions and run their own
simulation from scratch -- there is no shared input data. Each writes
its outputs (pattern matrices as `.npy`/`.csv`, and for
`fashion_model_animation.py`, the `.mp4`) next to itself, and the four
non-animation scripts also display the summary figure on screen
(`plt.show()`) unless a non-interactive Matplotlib backend is forced,
e.g. `MPLBACKEND=Agg`.

## Measured runtimes

Measured on an ordinary laptop (not a workstation), with Numba
installed. The three fast scripts were each timed alone, with no other
heavy process competing for CPU; see the note below for the two
extended scripts.

| Script | Measured runtime |
|---|---|
| `turing_pattern_vein_free.py` | ~1.5 min |
| `fashion_model.py` | ~1.3 min |
| `fashion_model_animation.py` | ~6 min |
| `fashion_model_extended.py` | ~27 min |
| `turing_pattern_vein_free_extended.py` | ~30 min |

The three fast scripts comfortably fit inside a single class session,
including live re-runs with changed parameters. Without Numba, expect
`fashion_model_animation.py` to run slower (its Gray-Scott loop is
Numba-accelerated when available); the other four scripts are plain
NumPy either way (see Dependencies above).

The two extended scripts were run concurrently with each other, so each
figure includes some mutual CPU contention -- running either one alone
would likely be a bit faster.

## License

MIT (see `LICENSE`).

## Citation

If you use this code, please cite:

Primel, J., Lefebvre, P., Mbaye, A. (2026). *Tricking Turing: How to Build a Geometric Checkerboard on Living Tissue [Simulation Code]*. Zenodo. https://doi.org/10.5281/zenodo.21944387
