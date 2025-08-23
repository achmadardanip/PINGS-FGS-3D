# PINGS: Physics-Informed Neural Network for Fast Generative Sampling

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Stars](https://img.shields.io/github/stars/username/pings?style=social)](https://github.com/username/pings)

Implementation of **PINGS (Physics-Informed Neural Network for Fast Generative Sampling)**, a framework to accelerate generative sampling by directly learning the solution of the reverse-time probability flow ODE with Physics-Informed Neural Networks (PINNs).

This repo contains two main experiments as described in the paper:

1. **Validation on a Canonical System**: The Damped Harmonic Oscillator (DHO)  
2. **Application**: Fast Generative Sampling of Non-Gaussian 3D Densities  

---

## 📂 Repository Structure

```

.
├── 3d\_gaussian.py               # PINGS on Non-Gaussian 3D densities
├── oscillator\_pinn.py            # PINN validation on damped oscillator ODE
├── pinn\_dho\_best.pt              # Saved checkpoint (DHO best model)
├── dho\_mse\_table.csv             # Evaluation results for DHO (MSE per ξ)
├── damped\_oscillator\_results.png # Visualization of PINN vs analytical DHO
├── Figure\_1.png                  # Example figure (generated results)
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation

````

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/achmadardanip/PINGS-Physics-Informed-Neural-Network-for-Fast-Generative-Sampling-of-Non-Gaussian-3D-Densities.git
cd to directory
python -m venv venv
source venv/bin/activate   # on Linux/Mac
venv\Scripts\activate      # on Windows

pip install -r requirements.txt
````

---

## 🚀 Usage

### 1. Damped Harmonic Oscillator (Validation)

Train PINN on the oscillator ODE and generate plots + MSE table:

```bash
python oscillator_pinn.py
```

Outputs:

* `damped_oscillator_results.png` : PINN prediction vs analytical solution
* `dho_mse_table.csv` : MSE for ξ = 0.1, 0.2, 0.3, 0.4

---

### 2. Fast Sampling on Non-Gaussian 3D Densities

Train PINGS on a 3D mixture of Gaussians and generate evaluation metrics:

```bash
python 3d_gaussian.py
```

Outputs:

* `pings_3d.pt` : trained PINGS model
* `pings_3d_stats.csv` : evaluation (MMD, mean/cov/skew/kurt MSE, runtime for 10k samples)
* `pings_3d_projection.png` : 2D projection of generated vs target samples

---

## 📊 Results

* **DHO**: PINN accurately matches analytical solutions with MSE on the order of 1e-5.
* **3D Gaussian Mixture**: PINGS achieves distribution alignment (mean/cov/skew/kurt) and generates 10,000 samples in \~20 ms on RTX 4090.

---

## 🛠️ Requirements

See [requirements.txt](requirements.txt), but main dependencies are:

* Python 3.9+
* PyTorch
* NumPy
* Matplotlib

---

## 📜 Citation

If you use this code, please cite the associated paper:

```
@article{pings2025,
  title={PINGS: Physics-Informed Neural Network for Fast Generative Sampling of Non-Gaussian 3D Densities},
  author={...},
  journal={...},
  year={2025}
}
```

