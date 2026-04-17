# Quantum-Informed Machine Learning for Predicting Spatiotemporal Chaos with Practical Quantum Advantage

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2507.19861)
[![Data](https://img.shields.io/badge/data-Zenodo-blue)](https://zenodo.org/records/16419086)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Quantum-Informed Machine Learning for Predicting Spatiotemporal Chaos with Practical Quantum Advantage"** at SCIENCE ADVANCES.

---

## Key Contributions

- **Novel Hybrid Architecture** — A new QIML framework that significantly improves long-term stability and statistical fidelity of chaotic system predictions.
- **Exceptional Efficiency** — Reduces data storage requirements by over two orders of magnitude compared to raw simulation data by leveraging the expressive power of quantum circuits.
- **Practical NISQ Application** — A scalable, hardware-compatible blueprint for integrating near-term (NISQ-era) quantum devices into classical scientific computing workflows.

---

## Citation

```bibtex
@article{
doi:10.1126/sciadv.aec5049,
author = {Maida Wang  and Xiao Xue  and Mingyang Gao  and Peter V. Coveney },
title = {Quantum-informed machine learning for predicting spatiotemporal chaos with practical quantum advantage},
journal = {Science Advances},
volume = {12},
number = {16},
pages = {eaec5049},
year = {2026},
doi = {10.1126/sciadv.aec5049},
URL = {https://www.science.org/doi/abs/10.1126/sciadv.aec5049}
}
```

---

## Repository Layout

| Path | Role |
|------|------|
| `script/` | Q-Prior training (`QPRIOR_*.py`) and QIML training (`train_q_*.py`) |
| `lib/` | Modules imported by training scripts (e.g. `vae_base.py`, `Koopman_2d.py`) |
| `data/` | Zenodo datasets (filenames must match those hard-coded in each script) |
| `postprocessing/` | Aggregation / analysis scripts |
| `visualise/` | Notebooks for figures (`visualization_*.ipynb`) |
| `models/`, `model/`, `mainmodel/` | Checkpoints (created by scripts when missing) |

> **Working directory note:** Training scripts use relative paths (`../data/`, `../lib`, `../models`). Always `cd script/` before running any training script — do **not** run them from the repository root.

---

## Installation

```bash
# Clone
git clone https://github.com/UCL-CCS/QIML.git
cd QIML

# Option A — conda
conda env create -f environment.yml
conda activate mlenv

# Option B — pip
pip install -r requirements.txt
```

---

## Data

Download files from [Zenodo](https://zenodo.org/records/16419086) and place them under `data/` using the exact filenames expected by each script (paths are case-sensitive on Linux):

| File | Used by |
|------|---------|
| `data/KS_data.npy` | `QPRIOR_ks_final.py`, `train_q_KS.py`, `postprocessing/modelload_down_ks.py` |
| `data/kf_2d_re1000_256_120seed.npy` | `QPRIOR_kf_final.py`, `train_q_kf.py` |
| `data/reduced_data.npy` | `QPRIOR_tcf_final.py` |
| `data/train_set_vxyz_s2_1_64.npy` | `train_q_TCF.py` |

---

## Usage

### 1. Train the Q-Prior

```bash
cd script
```

**Kuramoto–Sivashinsky**

Input: `../data/KS_data.npy` | Arguments: `--n_qubits`, `--epochs`, `--num_trajectories`

```bash
python QPRIOR_ks_final.py --n_qubits 10 --epochs 500 --num_trajectories 500
python QPRIOR_ks_final.py --help
```

**Kolmogorov flow**

Input: `../data/kf_2d_re1000_256_120seed.npy` | Arguments: `--n_qubits`, `--epochs`

```bash
python QPRIOR_kf_final.py --n_qubits 15 --epochs 300
python QPRIOR_kf_final.py --help
```

**Taylor–Green / TCF**

Input: `../data/reduced_data.npy` | Arguments: `--n_qubits`, `--epochs`

```bash
python QPRIOR_tcf_final.py --n_qubits 10 --epochs 500
python QPRIOR_tcf_final.py --help
```

---

### 2. Post-processing (KS example)

```bash
cd postprocessing
python modelload_down_ks.py --n_qubits 10 --num_trajectories 500
python modelload_down_ks.py --help
```

> **Path alignment required:** `QPRIOR_ks_final.py` writes checkpoints to `../models/model_ks_trajectories_128dim/`, while `modelload_down_ks.py` reads from `../models/model_ks_trajectories/`. Copy, symlink, or edit the scripts to align these paths before running the end-to-end KS pipeline.

---

### 3. Train the QIML Model

> **Note:** These scripts do not support `--system`, `--data_path`, or `--prior_path` arguments. Edit the hard-coded `np.load(...)` lines in each file to point to your Q-Prior array and data before running.

```bash
cd script
```

**KS** — `train_q_KS.py`

- Data: `../data/KS_data.npy`
- Q-Prior placeholder: `.../postprocessing/Q-Prior_ks_pdf_10.npy` ← replace with real path

```bash
python train_q_KS.py
```

**Kolmogorov flow** — `train_q_kf.py`

- Q-Prior: `../postprocessing/Q-Prior_kf_pdf_15_2.npy`
- Data: `../data/kf_2d_re1000_256_120seed.npy`

```bash
python train_q_kf.py
```

**TCF** — `train_q_TCF.py`

- Q-Prior: `../postprocessing/Q-Prior_iqm_0-9_32768.npy`
- Data: `../data/train_set_vxyz_s2_1_64.npy`

```bash
python train_q_TCF.py
```

**Classical baseline** — see `train_q_classical.py` for paths and settings.

---

### 4. Visualisation

Notebooks live under `visualise/`:

```
visualise/visualization_KS.ipynb
visualise/visualization_KF.ipynb
visualise/visualization_TCF.ipynb
```

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Contact

**Maida Wang** — maida.wang.24@ucl.ac.uk
**Xiao Xue** — x.xue@ucl.ac.uk

---

## Acknowledgements

We thank Professor Igor Mezic for valuable comments on an earlier version of this paper, and Dr. Marcello Benedetti for helpful feedback. We are grateful to Thomas M. Bickley and Angus Mingare for their careful reading and insightful comments. We also gratefully acknowledge IQM Quantum Computers for access to superconducting quantum processors used in hardware benchmarking, and the Leibniz Supercomputing Centre (LRZ) for access to the BEAST NVIDIA GPU cluster, which supported training of classical models and quantum circuit simulations.
