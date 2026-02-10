"""
analyze.py - Statistical analysis of latent state recovery

CONCEPTUAL OVERVIEW:
====================
This module tests the central hypothesis of state-space modeling:

    "Estimated latent states predict behavior better than raw observations"

If the Kalman filter is working correctly:
1. Correlation(estimated_state, RT) > Correlation(raw_EEG, RT)
2. The estimated state should approach the true latent state in accuracy

We quantify this with correlations and linear regression.

WHY THIS MATTERS:
- In real experiments, we never have access to the true latent state
- We only have noisy EEG and behavioral data
- State-space models let us extract a "cleaner" neural signal
- This cleaner signal is more predictive of what we care about (behavior)

This is the key insight that makes these models useful in neuroscience!
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple


def compute_correlations(
    latent_state_true: np.ndarray,
    latent_state_estimated: np.ndarray,
    eeg: np.ndarray,
    rt: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute correlations between neural measures and reaction time.
    
    This is the main analysis: does the estimated state predict RT
    better than raw EEG?
    
    Parameters
    ----------
    latent_state_true : np.ndarray
        True latent state (ground truth, not available in real data)
    latent_state_estimated : np.ndarray
        Kalman-filtered estimate of latent state
    eeg : np.ndarray
        Raw EEG observations
    rt : np.ndarray
        Reaction times
    
    Returns
    -------
    results : dict
        Nested dictionary with correlation results:
        - 'with_rt': correlations of each measure with reaction time
        - 'with_true_state': correlations with ground truth
        - 'improvement': quantification of filtering benefit
    
    Example
    -------
    >>> results = compute_correlations(x_true, x_est, eeg, rt)
    >>> print(f"RT prediction improved by {results['improvement']['rt_prediction']:.1%}")
    """
    results = {}
    
    # =========================================================================
    # 1. Correlations with Reaction Time
    # =========================================================================
    # This is what we really care about: predicting behavior
    
    r_true_rt, p_true_rt = stats.pearsonr(latent_state_true, rt)
    r_est_rt, p_est_rt = stats.pearsonr(latent_state_estimated, rt)
    r_eeg_rt, p_eeg_rt = stats.pearsonr(eeg, rt)
    
    results['with_rt'] = {
        'true_state': {'r': r_true_rt, 'p': p_true_rt},
        'estimated_state': {'r': r_est_rt, 'p': p_est_rt},
        'raw_eeg': {'r': r_eeg_rt, 'p': p_eeg_rt}
    }
    
    # =========================================================================
    # 2. Correlations with True Latent State
    # =========================================================================
    # How well did we recover the hidden state?
    
    r_est_true, p_est_true = stats.pearsonr(latent_state_estimated, latent_state_true)
    r_eeg_true, p_eeg_true = stats.pearsonr(eeg, latent_state_true)
    
    results['with_true_state'] = {
        'estimated_state': {'r': r_est_true, 'p': p_est_true},
        'raw_eeg': {'r': r_eeg_true, 'p': p_eeg_true}
    }
    
    # =========================================================================
    # 3. Quantify Improvement
    # =========================================================================
    # How much better is the estimated state vs raw EEG?
    
    # Improvement in RT prediction (using absolute values since correlations may be negative)
    improvement_rt = abs(r_est_rt) - abs(r_eeg_rt)
    
    # Improvement in state recovery
    improvement_state = abs(r_est_true) - abs(r_eeg_true)
    
    results['improvement'] = {
        'rt_prediction': improvement_rt,
        'state_recovery': improvement_state
    }
    
    return results


def run_regression_analysis(
    latent_state_estimated: np.ndarray,
    eeg: np.ndarray,
    rt: np.ndarray
) -> Dict[str, Dict]:
    """
    Run linear regression to quantify predictive power.
    
    We fit two models:
    1. RT ~ EEG (prediction from raw observations)
    2. RT ~ Estimated State (prediction from filtered state)
    
    Parameters
    ----------
    latent_state_estimated : np.ndarray
        Kalman-filtered state estimate
    eeg : np.ndarray
        Raw EEG observations
    rt : np.ndarray
        Reaction times
    
    Returns
    -------
    results : dict
        Regression results for each predictor:
        - 'slope': regression coefficient
        - 'intercept': intercept
        - 'r_squared': variance explained
        - 'p_value': significance
    
    Notes
    -----
    R-squared tells us how much variance in RT is explained by each predictor.
    A higher R² for estimated_state vs raw_eeg confirms the benefit of filtering.
    """
    results = {}
    
    # Regression: RT ~ Raw EEG
    slope_eeg, intercept_eeg, r_eeg, p_eeg, se_eeg = stats.linregress(eeg, rt)
    results['raw_eeg'] = {
        'slope': slope_eeg,
        'intercept': intercept_eeg,
        'r_squared': r_eeg ** 2,
        'r': r_eeg,
        'p_value': p_eeg,
        'std_error': se_eeg
    }
    
    # Regression: RT ~ Estimated State
    slope_est, intercept_est, r_est, p_est, se_est = stats.linregress(
        latent_state_estimated, rt
    )
    results['estimated_state'] = {
        'slope': slope_est,
        'intercept': intercept_est,
        'r_squared': r_est ** 2,
        'r': r_est,
        'p_value': p_est,
        'std_error': se_est
    }
    
    return results


def compute_autocorrelation(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Compute autocorrelation of a time series.
    
    Autocorrelation measures how much the signal at time t predicts the
    signal at time t+lag. This is important for understanding trial history.
    
    Parameters
    ----------
    x : np.ndarray
        Input time series
    max_lag : int
        Maximum lag to compute
    
    Returns
    -------
    autocorr : np.ndarray
        Autocorrelation at lags 0, 1, ..., max_lag
    
    Notes
    -----
    High autocorrelation in the latent state means brain states persist
    across trials - this is why trial history matters!
    """
    n = len(x)
    autocorr = np.zeros(max_lag + 1)
    
    x_centered = x - np.mean(x)
    var = np.var(x)
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            autocorr[lag] = np.mean(x_centered[:-lag] * x_centered[lag:]) / var
    
    return autocorr


def print_analysis_summary(
    correlations: Dict,
    regression: Dict
) -> None:
    """
    Print a formatted summary of the analysis results.
    
    Parameters
    ----------
    correlations : dict
        Output from compute_correlations()
    regression : dict
        Output from run_regression_analysis()
    """
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS: Does State-Space Modeling Improve RT Prediction?")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("1. CORRELATIONS WITH REACTION TIME")
    print("-" * 70)
    print(f"   Raw EEG -> RT:          r = {correlations['with_rt']['raw_eeg']['r']:+.3f} "
          f"(p = {correlations['with_rt']['raw_eeg']['p']:.4f})")
    print(f"   Estimated State -> RT:  r = {correlations['with_rt']['estimated_state']['r']:+.3f} "
          f"(p = {correlations['with_rt']['estimated_state']['p']:.4f})")
    print(f"   True State -> RT:       r = {correlations['with_rt']['true_state']['r']:+.3f} "
          f"(p = {correlations['with_rt']['true_state']['p']:.4f}) [ground truth]")
    
    print("\n" + "-" * 70)
    print("2. STATE RECOVERY ACCURACY")
    print("-" * 70)
    print(f"   Raw EEG <-> True State:      r = {correlations['with_true_state']['raw_eeg']['r']:+.3f}")
    print(f"   Estimated <-> True State:    r = {correlations['with_true_state']['estimated_state']['r']:+.3f}")
    
    print("\n" + "-" * 70)
    print("3. REGRESSION: VARIANCE EXPLAINED (R^2)")
    print("-" * 70)
    print(f"   Raw EEG predicting RT:       R^2 = {regression['raw_eeg']['r_squared']:.3f} "
          f"({regression['raw_eeg']['r_squared']*100:.1f}% of variance)")
    print(f"   Estimated State predicting RT: R^2 = {regression['estimated_state']['r_squared']:.3f} "
          f"({regression['estimated_state']['r_squared']*100:.1f}% of variance)")
    
    print("\n" + "-" * 70)
    print("4. IMPROVEMENT FROM KALMAN FILTERING")
    print("-" * 70)
    improvement = correlations['improvement']['rt_prediction']
    print(f"   Increase in |r| for RT prediction: {improvement:+.3f}")
    
    r2_improvement = (regression['estimated_state']['r_squared'] - 
                      regression['raw_eeg']['r_squared'])
    print(f"   Increase in R^2 for RT prediction:  {r2_improvement:+.3f} "
          f"({r2_improvement*100:+.1f} percentage points)")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    if improvement > 0:
        print("[+] The Kalman filter IMPROVED behavioral prediction!")
        print("    By filtering out observation noise, the estimated latent state")
        print("    captures the true neural dynamics that drive reaction time.")
    else:
        print("[-] Unexpectedly, the Kalman filter did not improve prediction.")
        print("    This might indicate model misspecification or very low noise.")
    print("=" * 70 + "\n")


# =============================================================================
# Quick test when run directly
# =============================================================================
if __name__ == "__main__":
    from simulate_data import simulate_experiment
    from kalman_filter import apply_kalman_filter
    
    # Generate data
    data = simulate_experiment(n_trials=200, seed=42)
    
    # Apply Kalman filter
    kf_results = apply_kalman_filter(
        y=data['eeg'],
        a=data['params']['a'],
        C=data['params']['C'],
        process_noise_var=data['params']['process_noise_std'] ** 2,
        observation_noise_var=data['params']['observation_noise_std'] ** 2
    )
    
    # Compute correlations
    correlations = compute_correlations(
        latent_state_true=data['latent_state'],
        latent_state_estimated=kf_results['smoothed_state'],
        eeg=data['eeg'],
        rt=data['rt']
    )
    
    # Run regression
    regression = run_regression_analysis(
        latent_state_estimated=kf_results['smoothed_state'],
        eeg=data['eeg'],
        rt=data['rt']
    )
    
    # Print summary
    print_analysis_summary(correlations, regression)
