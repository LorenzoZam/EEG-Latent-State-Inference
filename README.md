# EEG-Latent-State-Inference

This project implements a **linear state-space model** to demonstrate how latent neural dynamics can be recovered from noisy single-trial EEG observations using Kalman filtering. The estimated latent state is then shown to predict behavioral output (reaction time) more accurately than the raw EEG signal — illustrating the practical advantage of model-based neural signal processing.

---

## Motivation

Single-trial EEG analysis in cognitive neuroscience faces a fundamental challenge: the neural signal of interest is embedded in substantial measurement noise. Traditional approaches (e.g., trial averaging) discard trial-by-trial variability, which may carry meaningful information about cognitive fluctuations such as attention, arousal, or preparedness.

**State-space models** offer an alternative: they treat the observed EEG as a noisy measurement of a hidden (latent) brain state that evolves over trials according to known dynamics. By formalizing this process, we can apply optimal filtering (Kalman filter) to **separate signal from noise at the single-trial level**.

This project demonstrates the concept end-to-end: from generative model to inference to behavioral validation.

---

## Generative Model

The simulation follows a three-component linear generative model:

### 1. Latent Neural State (hidden)
$$x_t = a \cdot x_{t-1} + \epsilon_t^{(process)}$$

An AR(1) process representing a slowly fluctuating cognitive state (e.g., attentional readiness). The autoregressive parameter `a` (set to 0.9) controls temporal persistence — capturing the observation that brain states carry over across consecutive trials.

### 2. EEG Observation (measured)
$$y_t = C \cdot x_t + \epsilon_t^{(obs)}$$

The scalp-recorded EEG is modeled as a linear transformation of the latent state corrupted by observation noise (sensor noise, biological artifacts, unrelated neural activity).

### 3. Reaction Time (behavioral output)
$$RT_t = b - d \cdot x_t + \epsilon_t^{(behav)}$$

Behavior depends on the **true** latent state, not on measurement noise. Higher neural readiness (larger $x_t$) leads to faster responses (lower RT). This asymmetry is the reason that noise-filtered estimates should predict behavior better than raw observations.

---

## Approach

1. **Simulate** 200 trials from the generative model with known parameters
2. **Apply Kalman filtering** (forward pass) and **RTS smoothing** (backward pass) to the EEG observations to estimate the latent state
3. **Compare** the predictive power of raw EEG vs. the Kalman-estimated state on reaction time using Pearson correlations and linear regression

### Why Kalman Filtering?

The Kalman filter optimally combines two sources of information at each trial:
- **Prediction** from state dynamics: *"Based on the previous state, where should the current state be?"*
- **Observation** from the EEG: *"What does the current measurement tell us?"*

The relative weighting is governed by the **Kalman gain**, which adapts based on the noise structure — trusting observations more when they are precise, and relying on the dynamical model when observations are noisy.

The Rauch-Tung-Striebel (RTS) smoother extends this by incorporating future observations, yielding more accurate offline estimates suitable for post-hoc EEG analysis.

---

## Results

### State Recovery
| Measure | Correlation with True State |
|---------|-----------------------------|
| Raw EEG | r = 0.735 |
| Kalman filtered | r = 0.856 |
| Kalman smoothed | r = 0.892 |

### Behavioral Prediction
| Predictor | Correlation with RT | Variance Explained (R²) |
|-----------|---------------------|------------------------|
| Raw EEG | r = -0.603 | 36.4% |
| Estimated state (smoothed) | r = -0.714 | 50.9% |
| True state (ground truth) | r = -0.806 | 65.0% |

**Key finding**: Kalman smoothing recovers **+14.6 percentage points** of additional RT variance compared to raw EEG. This improvement arises from measurement error attenuation — removing observation noise reveals the underlying brain-behavior relationship more clearly.

---

## Project Structure

```
EEGpy/
├── main.py               # Entry point: runs the full pipeline with scientific commentary
├── simulate_data.py      # Generative model: latent state, EEG, and RT simulation
├── kalman_filter.py      # Kalman filter (manual) + RTS smoother + pykalman wrapper
├── analyze.py            # Statistical analysis: correlations, regression, autocorrelation
├── plot_results.py       # Visualization: state recovery, scatter 
└── README.md
```

Each module is self-contained and can be run independently for testing (`python <module>.py`).

---

## Installation & Usage

```bash

# Install dependencies
pip install numpy scipy matplotlib

# (Optional) for the pykalman wrapper
pip install pykalman

# Run the full analysis
python main.py
```

### Exploring Parameters

All model parameters are exposed in `main.py`. Modify these to explore different scenarios:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `a` | 0.9 | State persistence. Lower values → faster fluctuations, less filtering benefit |
| `observation_noise_std` | 2.0 | EEG noise level. Higher values → noisier EEG, larger filtering benefit |
| `process_noise_std` | 1.0 | State variability. Higher values → more dynamic latent state |
| `d` | 20.0 | State-to-RT coupling. Higher values → stronger brain-behavior link |
| `n_trials` | 200 | Sample size |

---

## References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Smith, A. C. & Brown, E. N. (2003). Estimating a state-space model from point process observations. *Neural Computation*, 15(5), 965–991.
- Vidaurre, D. et al. (2021). Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks. *Nature Communications*, 12(1), 1–13.

---

## License

MIT License — free to use for educational and research purposes.

