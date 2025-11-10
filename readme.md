# Quantum-Informed Machine Learning for Chaotic Systems

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/[Your_arXiv_ID_Here])
[![Data](https://img.shields.io/badge/data-Zenodo-blue)](https://zenodo.org/records/16419086)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code and implementation for the paper **"Quantum-Informed Machine Learning for Chaotic Systems"**.



## Key Contributions

* **Novel Hybrid Architecture:** We introduce and experimentally validate a new QIML framework that significantly improves the long-term stability and statistical fidelity of chaotic system predictions.
* **Exceptional Efficiency:** Our approach demonstrates remarkable parameter and memory efficiency, reducing data storage requirements by over two orders of magnitude compared to raw simulation data by leveraging the expressive power of quantum circuits.
* **Practical NISQ Application:** We provide a scalable and hardware-compatible blueprint for meaningfully integrating near-term (NISQ-era) quantum devices into classical scientific computing workflows to solve practical, complex problems.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{WangXueCoveney2025,
  title   = {Quantum-Informed Machine Learning for Chaotic Systems},
  author  = {Wang, Maida, Xue, Xiao and Coveney, Peter V.},
  journal = {arXiv},
  year    = {2025},
  volume  = {[Volume]},
  pages   = {[Pages]},
  doi     = {[DOI Here]}
}
```

## Installation

We recommend using a `conda` environment to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/UCL-CCS/QIML.git](https://github.com/UCL-CCS/QIML.git)
    cd QIML
    ```

2.  **Create and activate the conda environment:**
    ```bash
    conda create -n qiml python=3.9
    conda activate qiml
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The general workflow consists of training the QCBM to generate a prior, then training the main QIML model using this prior.

1.  **Download the Data:**
    All datasets are available on Zenodo. Please download the data and place it in the `data/` directory.

2.  **Train the Quantum Prior (QCBM):**
    (Example for the Kuramoto-Sivashinsky system)
    ```bash
    python train_qcbm.py --system KS --data_path data/ks_data.npy --output_path priors/ks_qprior.pkl
    ```

3.  **Train the QIML Model:**
    (Example for the Kuramoto-Sivashinsky system using the generated prior)
    ```bash
    python train_qiml.py --system KS --data_path data/ks_data.npy --prior_path priors/ks_qprior.pkl --output_path models/ks_qiml.pt
    ```

4.  **Evaluate the Model and Generate Figures:**
    (Example for generating figures for the KS system)
    ```bash
    python evaluate.py --system KS --model_path models/ks_qiml.pt
    ```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any questions, please feel free to contact:

* **Peter V. Coveney:** `p.v.coveney@ucl.ac.uk`
