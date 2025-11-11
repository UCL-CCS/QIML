# Quantum-Informed Machine Learning for Predicting Spatiotemporal Chaos

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/[Your_arXiv_ID_Here])
[![Data](https://img.shields.io/badge/data-Zenodo-blue)](https://zenodo.org/records/16419086)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code and implementation for the paper **"Quantum-Informed Machine Learning for Predicting Spatiotemporal Chaos"**.



## Key Contributions

* **Novel Hybrid Architecture:** We introduce and experimentally validate a new QIML framework that significantly improves the long-term stability and statistical fidelity of chaotic system predictions.
* **Exceptional Efficiency:** Our approach demonstrates remarkable parameter and memory efficiency, reducing data storage requirements by over two orders of magnitude compared to raw simulation data by leveraging the expressive power of quantum circuits.
* **Practical NISQ Application:** We provide a scalable and hardware-compatible blueprint for meaningfully integrating near-term (NISQ-era) quantum devices into classical scientific computing workflows to solve practical, complex problems.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{wang2025quantum,
  title={Quantum-Informed Machine Learning for Predicting Spatiotemporal Chaos},
  author={Wang, Maida and Xue, Xiao and Gao, Mingyang and Coveney, Peter V},
  journal={arXiv preprint arXiv:2507.19861},
  year={2025}
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

    **Recommended (Conda):**
    ```bash
    conda env create -f environment.yml
    conda activate mlenv
    ```

    **Alternative (Pip):**
    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** We recommend using the conda environment as it includes all system dependencies and ensures compatibility. The pip installation is provided as an alternative option.
## Usage

The general workflow consists of training the Q-Prior to generate a prior, then training the main QIML model using this prior.

1.  **Download the Data:**
    All datasets are available on Zenodo. Please download the data and place it in the `data/` directory.

2.  **Train the Quantum Prior (Q-Prior):**
    (Example for the Kuramoto-Sivashinsky system)
    ```bash
    python QPrior_ks_final.py --system KS --data_path data/ks_data.npy --output_path priors/ks_qprior.pkl
    ```

3.  **Train the QIML Model:**
    (Example for the Kuramoto-Sivashinsky system using the generated prior)
    ```bash
    python train_q_ks.py --system KS --data_path data/ks_data.npy --prior_path priors/ks_qprior.pkl --output_path models/ks_qiml.pt
    ```

4.  **Evaluate the Model and Generate Figures:**
    (Example for generating figures for the KS system)
    ```bash
    python visualization_ks.py --system KS --model_path models/ks_qiml.pt
    ```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any questions, please feel free to contact:

* **Maida Wang:** `maida.wang.24@ucl.ac.uk`


## Acknowledgements
We would like to thank Professor Igor Mezic for his valuable comments on an earlier version of this paper and Dr. Marcello Benedetti for his valuable feedback. We are also grateful to Thomas M. Bickley and Angus Mingare for their careful reading of the manuscript and insightful comments. We also gratefully acknowledge IQM Quantum Computers for providing access to superconducting quantum processors used in hardware benchmarking, and the Leibniz Supercomputing Centre (LRZ) for access to the BEAST NVDIA GPU cluster, which supported the training of classical models and quantum circuit simulations.

