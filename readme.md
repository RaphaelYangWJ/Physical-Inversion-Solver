# PIS: A Generalized Physical Inversion Solver (Anonymous Submission)

This repository contains the official PyTorch implementation for the paper: **"PIS: A Generalized Physical Inversion Solver for Arbitrary Sparse Observations via Set Conditioned Flow Matching"**.

> ⚠️ Note to Reviewers regarding Data and Weights:
>
> Due to the file size limits of the submission system (100MB) and the large scale of the generated physical fields, we cannot include the full pre-trained checkpoints and the complete dataset in this attachment.
>
> However, to ensure reproducibility, we provide: Complete data generation scripts for all three physical domains (Darcy, Helmholtz, SHM).
>
> **Full pre-trained weights and large-scale datasets will be publicly released upon acceptance.**

------

## Project Structure

Plaintext

```
.
├── data/
│   ├── darcy/          # Fortran & MATLAB scripts for Darcy Flow
│   ├── helmholtz/      # Downloader for Helmholtz dataset
│   └── shm/            # Python FEM scripts for SHM
├── models/
│   ├── model.py                # Set-Conditioned U-Net Backbone
│   ├── denoise_unet.py         # Sub-modules of Set-Conditioned U-Net Backbone
│   ├── encoder_net.py          # Sub-modules of Set-Conditioned U-Net Backbone
│   └── FM.py    # Flow Matching Framework
├── functions/
│   ├── data.py             # Dataloader module
│   ├── helper.py           # Helper functions for sampling and training
│   └── trainer.py          # Training Pipeline
├── Experiments/
│   ├── Field Sampler.py             # Sampling Notebook (Inversion Inference)
│   ├── Exp_Information.py           # Experiments: Information Theory
│   ├── Exp_Noise_Arb.py             # Experiments: Robustness to Noise
│   ├── Exp_testset_predicts.py      # Experiments: Testset Predicts
│   ├── Exp_UQ.py                    # Experiments: Uncertainty Quantification
│   └── outputs                      # Sampling results for Darcy Flow
├── output/               # Checkpoints saved here
├── main.py                 # Entry point for training
├── requirements.txt        # Dependencies
└── README.md
```

## Getting Started

### 1. Environment Setup

Create a virtual environment and install dependencies:

Bash

```
conda create -n pis_env python=3.9
conda activate pis_env
pip install -r requirements.txt
```

### 2. Reproducibility Pipeline

To fully reproduce the experiments from scratch, please follow the 4-step pipeline below.

#### Step 1: Generate Simulation Data (Raw Physics Fields)

The data generation process varies by physical domain. Please navigate to `data/` and run the specific scripts:

- **Subsurface Characterization (Darcy Flow):**
  - This module requires a **Fortran compiler** (for MODFLOW/MT3DMS) and **MATLAB**.
  - *Note: This generates raw hydraulic head and concentration fields and corresponding K fields.*\
- **Structural Health Monitoring (SHM):**
  - Run: the script in data/SHM
- **Wave-based Characterization (Helmholtz):**
  - This dataset can be download on Hugging Face: `camlab-ethz/Helmholtz`

#### Step 2: Data Processing

Convert the raw simulation outputs into the array format and saved as h5 file for subsequent training. This step normalizes coordinates and standardizes physical values.

- Run data preprocessor.ipynb for dataset preparations

#### Step 3: Training (PIS Framework)

To start training the model with the **Cosine-Annealed Sparsity Curriculum (CASC)**, run `main.py`. You can specify the task via config files.

Bash

```python
# Train PIS on the SHM task
python main.py
```

The script will automatically handle the curriculum scheduling defined in `training/casc.py`.

#### Step 4: Inference & Inversion

To evaluate the model and perform inversion on sparse observations, we provide an interactive Jupyter Notebook.

- Open `sampler.ipynb`.

- Load the model (or use the provided `tiny_debug.pt` for a sanity check).

  The notebook demonstrates how to:

  1. Sample sparse observations from ground truth.
  2. Run the ODE solver (Euler method, 20 NFEs).
  3. Visualize the inversion physical result.

------
