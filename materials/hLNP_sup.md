## 128x128 Resolution & Zero-Shot Generalization: Inversion Quality

| Model | Dense: 16384   | Sparse: 128    | Sparse: 64     | Sparse: 32     (Unseen Secnario) |
| ----- | -------------- | -------------- | -------------- | -------------------------------- |
| RMSE  | 0.024 ± 0.006  | 0.074 ± 0.033  | 0.114 ± 0.043  | 0.212 ± 0.072                    |
| SSIM  | 0.984 ± 0.008  | 0.935 ± 0.034  | 0.892 ± 0.049  | 0.790 ± 0.077                    |
| PSNR  | 32.765 ± 2.400 | 23.403 ± 3.764 | 19.500 ± 3.351 | 14.000 ± 3.058                   |

![128-Inversion_quality](images\128-Inversion_quality.png)

## 128x128 Resolution & Zero-Shot Generalization: Mutual Information

| Observation Sparsity | Wave-based Characterization |
| -------------------- | --------------------------- |
| 32                   | 21.261456                   |
| 64                   | 42.987223                   |
| 256                  | 48.991789                   |
| 16384                | 71.841272                   |

## 128x128 Resolution & Zero-Shot Generalization: Uncertainty Quantification

| Points | 1-sigma (%) | Exp 1-sigma (%) | 2-sigma (%) | Exp 2-sigma (%) | 3-sigma (%) | Exp 3-sigma (%) |
| ------ | ----------- | --------------- | ----------- | --------------- | ----------- | --------------- |
| 32     | 83.11       | 68.27           | 93.67       | 95.45           | 97.52       | 99.73           |
| 64     | 84.32       | 68.27           | 94.19       | 95.45           | 97.84       | 99.73           |
| 128    | 85.83       | 68.27           | 94.7        | 95.45           | 98.12       | 99.73           |

![128-Noise](images\128-Noise.png)

## 128x128 Resolution & Zero-Shot Generalization: Robustness to Noise

![128-Noise](images\128-Noise.png)

## Baseline Fairness: Interpolation vs. Raw Sparse Mask

We sincerely appreciate your careful scrutiny. Ensuring a strictly fair comparison is paramount to us. We completely agree that input modalities can significantly impact baseline performance. To eliminate any potential bias, we actually evaluated all grid-based baselines (e.g., FNO, PINN) under both interpolated and non-interpolated (raw sparse mask) conditions.

**Result Table for RMSE:** 

| Dataset    | Model         | Dense: 4096     | Sparse: 256     | Sparse: 64      | Sparse: 32      | Sparse: 16      | Sparse: 12      |
| ---------- | ------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Darcy Flow | PINN          | 0.045 ± 0.021 - | 0.045 ± 0.021 ↑ | 0.049 ± 0.020 ↑ | 0.054 ± 0.022 ↑ | 0.066 ± 0.029 ↑ | 0.075 ± 0.031 ↑ |
|            | FNO           | 0.020 ± 0.010 - | 0.097 ± 0.032 ↑ | 0.193 ± 0.087 ↑ | 0.300 ± 0.135 ↑ | 0.485 ± 0.234 ↑ | 0.537 ± 0.276 ↑ |
|            | NIO           | 0.024 ± 0.019 - | 0.068 ± 0.015 ↑ | 0.097 ± 0.017 ↑ | 0.115 ± 0.018 ↑ | 0.128 ± 0.020 ↑ | 0.136 ± 0.022 ↑ |
|            | Flow Matching | 0.046 ± 0.028 - | 0.079 ± 0.025 ↑ | 0.099 ± 0.025 ↑ | 0.112 ± 0.026 ↑ | 0.124 ± 0.026 ↑ | 0.132 ± 0.024 ↑ |
| Helmholtz  | PINN          | 0.157 ± 0.039 - | 0.228 ± 0.052 ↑ | 0.410 ± 0.094 ↑ | 0.512 ± 0.095   | 0.580 ± 0.102 ↓ | 0.589 ± 0.112 ↓ |
|            | FNO           | 0.023 ± 0.010 - | 0.448 ± 0.062 ↓ | 0.471 ± 0.070 ↓ | 0.532 ± 0.101 ↓ | 0.595 ± 0.145 ↓ | 0.617 ± 0.158 ↓ |
|            | NIO           | 0.090 ± 0.079 - | 0.393 ± 0.042 ↑ | 0.470 ± 0.048 ↑ | 0.527 ± 0.047 ↓ | 0.553 ± 0.055 ↓ | 0.561 ± 0.050 ↓ |
|            | Flow Matching | 0.090 ± 0.044 - | 0.410 ± 0.036 ↑ | 0.491 ± 0.044 ↑ | 0.551 ± 0.044 ↑ | 0.582 ± 0.039 ↓ | 0.582 ± 0.046 ↑ |
| SHM        | PINN          | 0.115 ± 0.010 - | 0.289 ± 0.074 ↑ | 0.703 ± 0.220 ↑ | 0.951 ± 0.278 ↓ | 1.128 ± 0.318 ↓ | 1.207 ± 0.342 ↓ |
|            | FNO           | 0.020 ± 0.002 - | 0.305 ± 0.061 ↑ | 0.607 ± 0.172 ↑ | 0.904 ± 0.308 ↓ | 1.076 ± 0.418 ↓ | 1.175 ± 0.443 ↓ |
|            | NIO           | 0.045 ± 0.043 - | 0.328 ± 0.075 ↑ | 0.548 ± 0.102 ↑ | 0.649 ± 0.106 ↑ | 0.745 ± 0.108 ↑ | 0.796 ± 0.106 ↑ |
|            | Flow Matching | 0.054 ± 0.055 - | 0.324 ± 0.125 ↑ | 0.561 ± 0.137 ↑ | 0.656 ± 0.128 ↑ | 0.777 ± 0.118 ↑ | 0.798 ± 0.118 ↑ |

### Darcy Flow for Baseline Fairness: Interpolation vs. Raw Sparse Mask

![Baselines_Darcy Flow](images\Baselines_Darcy_Flow.png)

### Helmholtz for Baseline Fairness: Interpolation vs. Raw Sparse Mask

![Baselines_Helmholtz](images\Baselines_Helmholtz.png)

### SHM for Baseline Fairness: Interpolation vs. Raw Sparse Mask

![Baselines_SHM](images\Baselines_SHM.png)

## Experiment update: Uncertainty Quantification and MI updates with coverage and PCAs

| Scenario   | Points | 1-sigma (%) | Exp 1-sigma (%) | 2-sigma (%) | Exp 2-sigma (%) | 3-sigma (%) | Exp 3-sigma (%) |
| ---------- | ------ | ----------- | --------------- | ----------- | --------------- | ----------- | --------------- |
| Darcy Flow | 12     | 71.11       | 68.27           | 92.66       | 95.45           | 96.59       | 99.73           |
| Darcy Flow | 16     | 73.46       | 68.27           | 94          | 95.45           | 97.36       | 99.73           |
| Darcy Flow | 32     | 76.97       | 68.27           | 93.1        | 95.45           | 96.97       | 99.73           |
| Helmholtz  | 12     | 83.31       | 68.27           | 93.31       | 95.45           | 97.07       | 99.73           |
| Helmholtz  | 16     | 79.08       | 68.27           | 92.14       | 95.45           | 96.79       | 99.73           |
| Helmholtz  | 32     | 83.13       | 68.27           | 93.75       | 95.45           | 97.54       | 99.73           |
| SHM        | 12     | 52.84       | 68.27           | 97.6        | 95.45           | 99.46       | 99.73           |
| SHM        | 16     | 54.04       | 68.27           | 97.78       | 95.45           | 99.47       | 99.73           |
| SHM        | 32     | 52.47       | 68.27           | 97.85       | 95.45           | 99.42       | 99.73           |

![coverage_barchart-2hvn](images\coverage_barchart-2hvn.png)

![calibration_curve-2hvn](images\calibration_curve-2hvn.png)

## Experiment update: Wall-Clock Inference Time Analysis and the Cost of the Set Encoder

Thanks for you advices as time consumption is crucial to show the engineering advantage in PIS. We have comprehensively benchmarked the wall-clock inference times of PIS and all baselines across 20 independent runs on the same hardware (a single NVIDIA RTX 5090 32GB GPU).

| Category                 | Model         | Preprocessing / Overhead   | Total Time   | UQ Capability |
| :----------------------- | :------------ | :------------------------- | :----------- | :------------ |
| **Deterministic**        | PINN          | None                       | **0.040 s**  | No            |
|                          | NIO           | Point Encoding             | **0.089 s**  | No            |
|                          | FNO           | Grid Interpolation         | **0.155 s**  | No            |
| **Probabilistic**        | Diffusion     | Standard DDPM (1000 steps) | **10.690 s** | Yes           |
|                          | DPS           | Explicit PDE Constraint    | **13.634 s** | Yes           |
|                          | Flow Matching | Grid Interpolation         | **3.035 s**  | Yes           |
| **Probabilistic (Ours)** | **PIS**       | **Set Encoder (+0.043 s)** | **3.078 s**  | **Yes **      |

As shown, processing the continuous off-grid sensors via the Set Encoder takes merely ~0.043 seconds (43 ms) per sample. Compared to the total ODE sampling time (~3.078 s), this overhead is virtually negligible (< 1.5%). This proves that we achieve our massive performance gains (by bypassing lossy grid interpolation) without introducing any meaningful computational bottleneck. We transparently acknowledge that deterministic operators (like PINN, FNO, and NIO) are significantly faster (~0.04 to ~0.15 s) than PIS. However, this is a fundamental and unavoidable mathematical trade-off: deterministic models only yield a single point estimate without any Uncertainty Quantification (UQ). Given the critical importance of UQ in safety-critical physical systems, an inference time of ~3 seconds per sample is highly competitive and well within the acceptable operational envelope.

## Experiment update for Toy 2D Bayesian: PIS Learned posterior is close to a reference Bayesian posterior

![PIS_gen](images\PIS_gen.png)

## Experiment update: Our rationale against Explicit PDE: Bridging the Academia-Engineering Gap

| Dataset    | Model | Sparse: 64 (RMSE) | Sparse: 32 (RMSE) | Sparse: 16 (RMSE) | **Sparse: 12 (RMSE) |
| ---------- | ----- | ----------------- | ----------------- | ----------------- | ------------------- |
| Darcy Flow | DPS   | 0.055             | 0.076             | 0.0830            | 0.087               |
| Helmholtz  | DPS   | 0.429             | 0.430             | 0.530             | 0.705               |
| SHM        | DPS   | 0.877             | 0.944             | 1.057             | 1.873               |

## Experiment update: Low-dimensional structure of the synthetic field generators

### Inversion Result Visualization - Generalization Beyond Specific Low-Dimensional Structures

![ADE](images\ADE.png)

### Experiment Result

| S/N  | Sparsity | SSIM     | SSIM STD | RMSE     | RMSE STD | PSNR     | PSNR STD | Lambda X | Lambda Y | KLE  |
| ---- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ---- |
| A1   | 4096     | 0.782646 | 0.032502 | 0.086028 | 0.026688 | 21.69241 | 2.620197 | 15       | 7.5      | 400  |
| A1   | 256      | 0.572627 | 0.05827  | 0.096552 | 0.033866 | 20.79179 | 2.940268 | 15       | 7.5      | 400  |
| A1   | 64       | 0.464971 | 0.064744 | 0.101654 | 0.03352  | 20.27258 | 2.687875 | 15       | 7.5      | 400  |
| A1   | 32       | 0.395738 | 0.082495 | 0.105546 | 0.03576  | 19.98586 | 2.840024 | 15       | 7.5      | 400  |
| A1   | 16       | 0.334311 | 0.086007 | 0.103181 | 0.035896 | 20.19114 | 2.836654 | 15       | 7.5      | 400  |
| A1   | 12       | 0.335757 | 0.084962 | 0.105158 | 0.037443 | 20.05097 | 2.929751 | 15       | 7.5      | 400  |
| A2   | 4096     | 0.669326 | 0.047378 | 0.162783 | 0.056104 | 16.29174 | 3.138394 | 22.5     | 11.25    | 400  |
| A2   | 256      | 0.519747 | 0.075629 | 0.174354 | 0.062447 | 15.72505 | 3.199391 | 22.5     | 11.25    | 400  |
| A2   | 64       | 0.449912 | 0.087915 | 0.178087 | 0.063113 | 15.51897 | 3.117503 | 22.5     | 11.25    | 400  |
| A2   | 32       | 0.403144 | 0.096507 | 0.178543 | 0.063513 | 15.50938 | 3.175096 | 22.5     | 11.25    | 400  |
| A2   | 16       | 0.352963 | 0.103478 | 0.175761 | 0.066547 | 15.71135 | 3.348913 | 22.5     | 11.25    | 400  |
| A2   | 12       | 0.341092 | 0.100872 | 0.179106 | 0.067845 | 15.54741 | 3.351514 | 22.5     | 11.25    | 400  |
| A3   | 4096     | 0.822924 | 0.089587 | 0.048435 | 0.024192 | 27.20212 | 3.973626 | 30       | 15       | 400  |
| A3   | 256      | 0.623759 | 0.07249  | 0.059499 | 0.028501 | 25.37204 | 3.892185 | 30       | 15       | 400  |
| A3   | 64       | 0.500846 | 0.099145 | 0.064597 | 0.029123 | 24.58505 | 3.762182 | 30       | 15       | 400  |
| A3   | 32       | 0.415083 | 0.108477 | 0.067256 | 0.032407 | 24.33784 | 4.030288 | 30       | 15       | 400  |
| A3   | 16       | 0.330315 | 0.120789 | 0.067816 | 0.032342 | 24.27435 | 4.027621 | 30       | 15       | 400  |
| A3   | 12       | 0.329369 | 0.118728 | 0.071784 | 0.034736 | 23.75677 | 3.959212 | 30       | 15       | 400  |
| A4   | 4096     | 0.787427 | 0.065144 | 0.079659 | 0.038515 | 22.87772 | 4.020177 | 45       | 22.5     | 400  |
| A4   | 256      | 0.593132 | 0.070609 | 0.089233 | 0.049325 | 22.20267 | 4.712467 | 45       | 22.5     | 400  |
| A4   | 64       | 0.48988  | 0.09132  | 0.093139 | 0.049167 | 21.72545 | 4.493144 | 45       | 22.5     | 400  |
| A4   | 32       | 0.417318 | 0.098005 | 0.096479 | 0.051448 | 21.46407 | 4.613099 | 45       | 22.5     | 400  |
| A4   | 16       | 0.35146  | 0.111004 | 0.092718 | 0.053341 | 22.00602 | 5.047677 | 45       | 22.5     | 400  |
| A4   | 12       | 0.344513 | 0.108737 | 0.095064 | 0.054531 | 21.7287  | 4.861275 | 45       | 22.5     | 400  |
| A5   | 4096     | 0.721335 | 0.077455 | 0.100777 | 0.05182  | 21.02471 | 4.50085  | 60       | 30       | 400  |
| A5   | 256      | 0.567531 | 0.069625 | 0.106621 | 0.063045 | 20.93209 | 5.360062 | 60       | 30       | 400  |
| A5   | 64       | 0.465005 | 0.086015 | 0.110856 | 0.063898 | 20.45697 | 5.029322 | 60       | 30       | 400  |
| A5   | 32       | 0.40516  | 0.095916 | 0.111223 | 0.065669 | 20.50273 | 5.14277  | 60       | 30       | 400  |
| A5   | 16       | 0.338494 | 0.09588  | 0.109944 | 0.068723 | 20.76702 | 5.468386 | 60       | 30       | 400  |
| A5   | 12       | 0.337145 | 0.09846  | 0.110456 | 0.06678  | 20.69537 | 5.477519 | 60       | 30       | 400  |
| B1   | 4096     | 0.741564 | 0.094328 | 0.096163 | 0.039465 | 21.01731 | 3.503639 | 30       | 15       | 50   |
| B1   | 256      | 0.651432 | 0.073916 | 0.102712 | 0.046618 | 20.59374 | 3.894666 | 30       | 15       | 50   |
| B1   | 64       | 0.586144 | 0.082402 | 0.105707 | 0.04973  | 20.41236 | 4.062178 | 30       | 15       | 50   |
| B1   | 32       | 0.516966 | 0.10051  | 0.106893 | 0.051927 | 20.38329 | 4.253439 | 30       | 15       | 50   |
| B1   | 16       | 0.417534 | 0.112562 | 0.106497 | 0.055028 | 20.52526 | 4.464489 | 30       | 15       | 50   |
| B1   | 12       | 0.399173 | 0.116974 | 0.106892 | 0.051118 | 20.3987  | 4.323262 | 30       | 15       | 50   |
| B2   | 4096     | 0.779815 | 0.07854  | 0.079526 | 0.033202 | 22.67166 | 3.511677 | 30       | 15       | 100  |
| B2   | 256      | 0.668925 | 0.060013 | 0.086123 | 0.04065  | 22.18034 | 3.995669 | 30       | 15       | 100  |
| B2   | 64       | 0.560085 | 0.094973 | 0.090915 | 0.040852 | 21.62528 | 3.784346 | 30       | 15       | 100  |
| B2   | 32       | 0.470118 | 0.109842 | 0.094377 | 0.043318 | 21.33554 | 3.881223 | 30       | 15       | 100  |
| B2   | 16       | 0.376592 | 0.107464 | 0.092052 | 0.044825 | 21.62103 | 4.005565 | 30       | 15       | 100  |
| B2   | 12       | 0.366485 | 0.122474 | 0.093785 | 0.043586 | 21.47908 | 4.182744 | 30       | 15       | 100  |
| B3   | 4096     | 0.806166 | 0.075156 | 0.062642 | 0.025627 | 24.72564 | 3.458172 | 30       | 15       | 200  |
| B3   | 256      | 0.647612 | 0.073637 | 0.070498 | 0.033034 | 23.87486 | 3.859344 | 30       | 15       | 200  |
| B3   | 64       | 0.523724 | 0.101303 | 0.07485  | 0.036089 | 23.43299 | 4.067132 | 30       | 15       | 200  |
| B3   | 32       | 0.45378  | 0.106408 | 0.078702 | 0.035431 | 22.8531  | 3.682003 | 30       | 15       | 200  |
| B3   | 16       | 0.344571 | 0.125472 | 0.07819  | 0.035878 | 23.03611 | 4.117768 | 30       | 15       | 200  |
| B3   | 12       | 0.352446 | 0.122016 | 0.079084 | 0.037938 | 22.9575  | 4.107986 | 30       | 15       | 200  |
| B5   | 4096     | 0.768117 | 0.08943  | 0.050159 | 0.021497 | 26.67045 | 3.449108 | 30       | 15       | 800  |
| B5   | 256      | 0.578779 | 0.074136 | 0.060321 | 0.028384 | 25.21674 | 3.823041 | 30       | 15       | 800  |
| B5   | 64       | 0.474844 | 0.090073 | 0.065568 | 0.030514 | 24.46424 | 3.732076 | 30       | 15       | 800  |
| B5   | 32       | 0.399849 | 0.101516 | 0.067861 | 0.03223  | 24.24371 | 3.972828 | 30       | 15       | 800  |
| B5   | 16       | 0.313525 | 0.111339 | 0.069654 | 0.031823 | 23.96559 | 3.85938  | 30       | 15       | 800  |
| B5   | 12       | 0.318622 | 0.113604 | 0.071022 | 0.030796 | 23.74726 | 3.777274 | 30       | 15       | 800  |