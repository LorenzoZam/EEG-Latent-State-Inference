"""
kalman_filter.py - Apply Kalman filtering to recover latent state from EEG

CONCEPTUAL OVERVIEW:
====================
The Kalman filter is an algorithm that optimally estimates hidden states from
noisy observations. It's the workhorse of state-space modeling.

WHY DO WE NEED THIS?
- Raw EEG (y_t) is corrupted by measurement noise
- We want to recover the true neural state (x_t) that drives behavior
- The Kalman filter uses our knowledge of the system dynamics to "denoise"

HOW IT WORKS (Intuition):
1. PREDICT: Use the state equation to predict where the state should be
   x_predicted = a * x_previous

2. UPDATE: Compare prediction to observation, adjust based on uncertainty
   x_filtered = x_predicted + K * (y_observed - C * x_predicted)
   
   where K is the "Kalman gain" - how much to trust the observation vs prediction

3. SMOOTH (optional): Use future observations to refine past estimates
   This gives even better estimates, especially for offline analysis

KEY INSIGHT:
The Kalman filter balances two sources of information:
- Our model of how the state evolves (the dynamics)
- What we actually observed (the EEG)

If observations are very noisy → trust the dynamics more
If dynamics are unpredictable → trust the observations more
"""

import numpy as np
from typing import Dict, Tuple


def apply_kalman_filter(
    y: np.ndarray,
    a: float = 0.9,
    C: float = 1.0,
    process_noise_var: float = 1.0,
    observation_noise_var: float = 4.0,
    initial_state_mean: float = 0.0,
    initial_state_var: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Apply Kalman filter/smoother to estimate latent state from EEG observations.
    
    This implements the classical Kalman filter for a 1D linear system:
        State equation:       x_t = a * x_{t-1} + process_noise
        Observation equation: y_t = C * x_t + observation_noise
    
    Parameters
    ----------
    y : np.ndarray
        Observed EEG signal (n_trials,)
    a : float
        State transition coefficient (should match simulation)
    C : float
        Observation gain
    process_noise_var : float
        Variance of process noise (Q in Kalman notation)
    observation_noise_var : float
        Variance of observation noise (R in Kalman notation)
    initial_state_mean : float
        Prior mean for the initial state
    initial_state_var : float
        Prior variance for the initial state
    
    Returns
    -------
    results : dict
        Dictionary with:
        - 'filtered_state': Forward-filtered estimates (uses only past data)
        - 'smoothed_state': Smoothed estimates (uses all data, better accuracy)
        - 'filtered_cov': Uncertainty of filtered estimates
        - 'smoothed_cov': Uncertainty of smoothed estimates
    
    Notes
    -----
    We implement both filtering and smoothing:
    - FILTERING: x_t|y_{1:t} - estimate using only past+current observations
    - SMOOTHING: x_t|y_{1:T} - estimate using ALL observations (past and future)
    
    Smoothing gives better estimates but requires the full dataset.
    In real-time applications, you'd use filtering only.
    For offline analysis (like EEG studies), smoothing is preferred.
    """
    n_trials = len(y)
    
    # =========================================================================
    # FORWARD PASS: Filtering
    # =========================================================================
    # At each time step, we predict then update
    
    # Storage arrays
    x_filtered = np.zeros(n_trials)    # Filtered state estimates
    P_filtered = np.zeros(n_trials)    # Filtered covariances (uncertainty)
    x_predicted = np.zeros(n_trials)   # One-step predictions
    P_predicted = np.zeros(n_trials)   # Prediction covariances
    
    # Initialize with prior
    x_filtered[0] = initial_state_mean
    P_filtered[0] = initial_state_var
    
    for t in range(n_trials):
        # =====================================================================
        # PREDICT STEP
        # "Where do we think the state is, based on the dynamics?"
        # =====================================================================
        if t == 0:
            # First trial: use prior
            x_pred = initial_state_mean
            P_pred = initial_state_var
        else:
            # Predict using state equation: x_t = a * x_{t-1}
            x_pred = a * x_filtered[t - 1]
            
            # Prediction uncertainty grows due to process noise
            P_pred = (a ** 2) * P_filtered[t - 1] + process_noise_var
        
        x_predicted[t] = x_pred
        P_predicted[t] = P_pred
        
        # =====================================================================
        # UPDATE STEP
        # "How do we adjust our prediction based on what we observed?"
        # =====================================================================
        
        # Innovation: difference between observed and predicted observation
        # This is the "surprise" in the data
        innovation = y[t] - C * x_pred
        
        # Innovation variance: how uncertain are we about this surprise?
        S = (C ** 2) * P_pred + observation_noise_var
        
        # Kalman gain: how much to trust the observation vs the prediction
        # K close to 0: trust prediction (observations very noisy)
        # K close to 1: trust observation (dynamics very noisy)
        K = (P_pred * C) / S
        
        # Update state estimate
        x_filtered[t] = x_pred + K * innovation
        
        # Update covariance (uncertainty shrinks after seeing data)
        P_filtered[t] = (1 - K * C) * P_pred
    
    # =========================================================================
    # BACKWARD PASS: Smoothing (Rauch-Tung-Striebel smoother)
    # =========================================================================
    # Go backwards through time, refining estimates using future observations
    
    x_smoothed = np.zeros(n_trials)
    P_smoothed = np.zeros(n_trials)
    
    # Last time point: smoothed = filtered (no future data)
    x_smoothed[-1] = x_filtered[-1]
    P_smoothed[-1] = P_filtered[-1]
    
    # Backward pass
    for t in range(n_trials - 2, -1, -1):
        # Smoother gain: how to combine filtered estimate with future info
        J = (P_filtered[t] * a) / P_predicted[t + 1]
        
        # Smoothed estimate: blend filtered with information from future
        x_smoothed[t] = x_filtered[t] + J * (x_smoothed[t + 1] - x_predicted[t + 1])
        
        # Smoothed covariance
        P_smoothed[t] = P_filtered[t] + (J ** 2) * (P_smoothed[t + 1] - P_predicted[t + 1])
    
    return {
        'filtered_state': x_filtered,
        'smoothed_state': x_smoothed,
        'filtered_cov': P_filtered,
        'smoothed_cov': P_smoothed
    }


def apply_kalman_filter_pykalman(
    y: np.ndarray,
    a: float = 0.9,
    C: float = 1.0,
    process_noise_var: float = 1.0,
    observation_noise_var: float = 4.0
) -> Dict[str, np.ndarray]:
    """
    Apply Kalman filter using the pykalman library.
    
    This is an alternative implementation using a well-tested library.
    Produces the same results as our manual implementation.
    
    Parameters
    ----------
    y : np.ndarray
        Observed EEG signal
    a, C : float
        Model parameters
    process_noise_var, observation_noise_var : float
        Noise variances
    
    Returns
    -------
    results : dict
        Same structure as apply_kalman_filter()
    
    Notes
    -----
    pykalman uses slightly different notation:
    - transition_matrices = our 'a'
    - observation_matrices = our 'C'
    - transition_covariance = Q = process_noise_var
    - observation_covariance = R = observation_noise_var
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        raise ImportError(
            "pykalman not installed. Install with: pip install pykalman\n"
            "Or use apply_kalman_filter() for the manual implementation."
        )
    
    # Create Kalman filter object
    kf = KalmanFilter(
        transition_matrices=[a],           # State transition
        observation_matrices=[C],          # Observation matrix
        transition_covariance=[process_noise_var],     # Process noise Q
        observation_covariance=[observation_noise_var], # Observation noise R
        initial_state_mean=[0],
        initial_state_covariance=[[1.0]]
    )
    
    # Reshape observations for pykalman (needs 2D input)
    y_2d = y.reshape(-1, 1)
    
    # Run filter (forward only)
    filtered_state_means, filtered_state_covs = kf.filter(y_2d)
    
    # Run smoother (forward + backward)
    smoothed_state_means, smoothed_state_covs = kf.smooth(y_2d)
    
    return {
        'filtered_state': filtered_state_means.flatten(),
        'smoothed_state': smoothed_state_means.flatten(),
        'filtered_cov': filtered_state_covs.flatten(),
        'smoothed_cov': smoothed_state_covs.flatten()
    }


# =============================================================================
# Quick test when run directly
# =============================================================================
if __name__ == "__main__":
    # Import simulation module
    from simulate_data import simulate_experiment
    
    # Generate test data
    data = simulate_experiment(n_trials=200, seed=42)
    
    # Apply Kalman filter (manual implementation)
    results = apply_kalman_filter(
        y=data['eeg'],
        a=data['params']['a'],
        C=data['params']['C'],
        process_noise_var=data['params']['process_noise_std'] ** 2,
        observation_noise_var=data['params']['observation_noise_std'] ** 2
    )
    
    # Compute correlation with true state
    from scipy.stats import pearsonr
    
    corr_filtered, _ = pearsonr(data['latent_state'], results['filtered_state'])
    corr_smoothed, _ = pearsonr(data['latent_state'], results['smoothed_state'])
    corr_raw_eeg, _ = pearsonr(data['latent_state'], data['eeg'])
    
    print("=" * 60)
    print("KALMAN FILTER RECOVERY ACCURACY")
    print("=" * 60)
    print(f"\nCorrelation with TRUE latent state:")
    print(f"  Raw EEG:         r = {corr_raw_eeg:.3f}")
    print(f"  Filtered state:  r = {corr_filtered:.3f}")
    print(f"  Smoothed state:  r = {corr_smoothed:.3f}")
    print(f"\nSmoothing improves recovery because it uses future information.")
    print("=" * 60)
