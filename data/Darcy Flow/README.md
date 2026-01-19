# Subsurface Flow Training Dataset Generation

This project is a MATLAB-based framework designed to generate high-fidelity training datasets for subsurface flow and contaminant transport simulations. It utilizes **MODFLOW** and **MT3DMS** to simulate groundwater flow and solute transport in heterogeneous porous media. The heterogeneity of the hydraulic conductivity field is modeled using Karhunen-Loève (KL) expansion.

The primary output is a large-scale HDF5 dataset containing hydraulic head and concentration fields over time, suitable for training machine learning models (e.g., surrogate models, inversion networks).

## Features

- **Random Field Generation:** Generates spatially correlated random hydraulic conductivity fields using KL expansion.
- **Parallel Computing:** Leverages MATLAB's `parfor` to run multiple simulations concurrently, maximizing computational efficiency.
- **Coupled Simulation:** Automates the execution of MODFLOW (flow) and MT3DMS (transport) solvers.
- **Efficient Data Storage:** Aggregates simulation results (concentration snapshots and steady-state heads) into a single, compressed HDF5 file.
- **Visualization:** Includes tools to inspect and visualize the generated data.

## File Structure

- **`run_Gaussian_simulaiton.m`**: The main entry point script. It configures parameters, manages the simulation loop, and saves the final dataset.
- **`model_H.m`**: The core simulation function. It prepares input files, executes the solvers, and extracts results for a single sample.
- **`visualize_sample1.m`**: A utility script to read the generated HDF5 file and visualize the results (concentration evolution and head distribution) for the first sample.
- **`high_fidelity/`**: Contains the template simulation files (MODFLOW/MT3DMS inputs) and batch scripts.
  - **`parallel_1/`**: The base directory copied for parallel simulations.
  - **`copyexample.m`**: Helper script to duplicate/clean up simulation directories.
- **`Utilities/`**: Contains helper functions for data I/O and mathematical operations.
  - **`generate_kl_random_field.m`**: Generates KL expansion coefficients and basis functions.
  - **`readMT3D.m`**: Reads concentration data from MT3DMS output files (`.UCN`).
  - **`readDat.m`**: Reads hydraulic head data from MODFLOW output files (`.hed`).

## Prerequisites

- **MATLAB** (with Parallel Computing Toolbox recommended for speed).
- **MODFLOW & MT3DMS**: The project relies on external batch scripts (`modflow.bat`, `mt3dms5b.bat`) located in `high_fidelity/parallel_1/` to run the solvers. Ensure these are correctly configured for your environment.

## Usage Guide

1.  **Configure Parameters:**
    Open `run_Gaussian_simulaiton.m` and adjust the simulation parameters as needed:
    ```matlab
    Ne       = 200;   % Number of samples to generate
    Lx       = 63;    % Domain length in X
    Ly       = 63;    % Domain length in Y
    kl_num   = 400;   % Number of KL terms for random field generation
    ```

2.  **Run the Simulation:**
    Execute the `run_Gaussian_simulaiton.m` script in MATLAB.
    ```matlab
    >> run_Gaussian_simulaiton
    ```
    The script will:
    - Generate random fields.
    - Create parallel simulation environments.
    - Run flow and transport simulations for `Ne` samples.
    - Save the results to `C_H_all_full_field_results.h5`.
    - Clean up temporary directories.

3.  **Visualize Results:**
    After the simulation completes, use `visualize_sample1.m` to inspect the output:
    ```matlab
    >> visualize_sample1
    ```
    This will display the concentration evolution over 20 time steps and the final hydraulic head distribution for the first sample. It also saves the visualization as `.png` and `.fig` files.

## Output Format

The results are saved in **`C_H_all_full_field_results.h5`** with the following structure:

- **/concentration_data**: `[64 x 64 x 20 x Ne]` (Single precision)
  - 20 time steps of concentration fields.
- **/head_data**: `[64 x 64 x Ne]` (Single precision)
  - Steady-state hydraulic head fields.
- **/time_steps**: `[20 x 1]` (Double precision)
  - The simulation times corresponding to the concentration snapshots (e.g., 50, 100, ..., 1000).

## Notes

- Ensure that the path to the `Utilities` folder is correctly added (handled automatically in the main script).
- The simulation assumes a grid size of 64x64. If you change `Lx`, `Ly`, or the grid resolution, ensure that the MODFLOW/MT3DMS input files in `high_fidelity/parallel_1` are updated accordingly.
