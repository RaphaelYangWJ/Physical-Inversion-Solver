## 1. Experiment update: Coverage Curves for Uncertainty Quantification

| Scenario   | Points | 1-sigma (%) | Exp 1-sigma (%) | 2-sigma (%) | Exp 2-sigma (%) | 3-sigma (%) | Exp 3-sigma (%) |
| :--------- | :----: | :---------: | :-------------: | :---------: | :-------------: | :---------: | :-------------: |
| Darcy Flow |   12   |    71.11    |      68.27      |    92.66    |      95.45      |    96.59    |      99.73      |
| Darcy Flow |   16   |    73.46    |      68.27      |     94      |      95.45      |    97.36    |      99.73      |
| Darcy Flow |   32   |    76.97    |      68.27      |    93.1     |      95.45      |    96.97    |      99.73      |
| Helmholtz  |   12   |    83.31    |      68.27      |    93.31    |      95.45      |    97.07    |      99.73      |
| Helmholtz  |   16   |    79.08    |      68.27      |    92.14    |      95.45      |    96.79    |      99.73      |
| Helmholtz  |   32   |    83.13    |      68.27      |    93.75    |      95.45      |    97.54    |      99.73      |
| SHM        |   12   |    52.84    |      68.27      |    97.6     |      95.45      |    99.46    |      99.73      |
| SHM        |   16   |    54.04    |      68.27      |    97.78    |      95.45      |    99.47    |      99.73      |
| SHM        |   32   |    52.47    |      68.27      |    97.85    |      95.45      |    99.42    |      99.73      |

Coverage Chart

![coverage_barchart-2hvn](images\coverage_barchart-2hvn.png)

UQ Calibration visualization

![calibration_curve-2hvn](images\calibration_curve-2hvn.png)

## 2. Experiment update: Kozachenko-Leonenko estimator with PCAs at d=16

| Sparsity | Subsurface Characterization | Wave-based Characterization | Structural Health Monitoring |
| -------- | --------------------------- | --------------------------- | ---------------------------- |
| 16       | 23.814529                   | 10.488970                   | 100.273170                   |
| 32       | 39.413468                   | 19.522221                   | 107.557562                   |
| 64       | 49.314401                   | 43.305902                   | 128.528124                   |
| 256      | 62.879528                   | 60.238919                   | 129.515975                   |

![MI-2hvn](images\MI-2hvn.png)

## 3. Experiment update: CASC for Baselines

| Model         | Scenario   | Dense: 4096     | Sparse: 256     | Sparse: 64      | Sparse: 32      | Sparse: 16      | Sparse: 12      |
| ------------- | ---------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| PINN          | Darcy Flow | 0.048 ± 0.025 ↓ | 0.388 ± 0.025 ↓ | 0.444 ± 0.026 ↓ | 0.456 ± 0.026 ↓ | 0.462 ± 0.027 ↓ | 0.464 ± 0.027 ↓ |
| PINN          | Helmholtz  | 0.226 ± 0.044 ↓ | 0.256 ± 0.039 ↑ | 0.299 ± 0.036 ↑ | 0.315 ± 0.038 ↑ | 0.322 ± 0.038 ↑ | 0.325 ± 0.037 ↑ |
| PINN          | SHM        | 0.175 ± 0.022 ↓ | 0.525 ± 0.082 ↑ | 0.655 ± 0.058↑  | 0.680 ± 0.049 ↑ | 0.700 ± 0.045 ↑ | 0.697 ± 0.042 ↑ |
| FNO           | Darcy Flow | 0.032 ± 0.019 ↓ | 0.547 ± 0.027 ↑ | 0.559 ± 0.028 ↑ | 0.561 ± 0.028 ↑ | 0.562 ± 0.028 ↑ | 0.563 ± 0.028 ↑ |
| FNO           | Helmholtz  | 0.060 ± 0.020 ↓ | 0.374 ± 0.063 ↑ | 0.368 ± 0.051 ↑ | 0.371 ± 0.045 ↑ | 0.371 ± 0.040 ↑ | 0.374 ± 0.039 ↑ |
| FNO           | SHM        | 0.071 ± 0.010 ↓ | 0.837 ± 0.083 ↑ | 0.812 ± 0.075 ↑ | 0.807 ± 0.074 ↑ | 0.804 ± 0.074 ↑ | 0.803 ± 0.074 ↑ |
| NIO           | Darcy Flow | 0.038 ± 0.023 ↓ | 0.046 ± 0.022 ↑ | 0.068 ± 0.019 ↑ | 0.098 ± 0.020 ↑ | 0.126 ± 0.021 ↑ | 0.137 ± 0.021 ↑ |
| NIO           | Helmholtz  | 0.213 ± 0.122 ↓ | 0.567 ± 0.052 ↓ | 0.627 ± 0.054 ↓ | 0.607 ± 0.056 ↓ | 0.567 ± 0.042 ↓ | 0.553 ± 0.043 ↓ |
| NIO           | SHM        | 0.099 ± 0.058 ↓ | 0.953 ± 0.111 ↑ | 1.019 ± 0.107 ↑ | 1.045 ± 0.111 ↓ | 1.061 ± 0.105 ↓ | 1.065 ± 0.120 ↓ |
| Flow Matching | Darcy Flow | 0.071 ± 0.026 ↓ | 0.074 ± 0.026 ↑ | 0.081 ± 0.024 ↑ | 0.091 ± 0.023 ↑ | 0.097 ± 0.022 ↑ | 0.101 ± 0.027 ↑ |
| Flow Matching | Helmholtz  | 0.183 ± 0.044 ↓ | 0.617 ± 0.037 ↓ | 0.632 ± 0.042 ↓ | 0.612 ± 0.044 ↓ | 0.595 ± 0.049 ↓ | 0.599 ± 0.051 ↓ |
| Flow Matching | SHM        | 0.085 ± 0.072 ↓ | 0.108 ± 0.098 ↓ | 0.323 ± 0.174 ↑ | 0.562 ± 0.167 ↑ | 0.699 ± 0.156 ↑ | 0.759 ± 0.147 ↑ |

Darcy Flow

![Baselines_Darcy Flow-2hvn](images\Baselines_Darcy Flow-2hvn.png)

Helmholtz

![Baselines_Helmholtz-2hvn](images\Baselines_Helmholtz-2hvn.png)

SHM![Baselines_SHM-2hvn](images\Baselines_SHM-2hvn.png)