"""
==============================================================================
STATE-SPACE MODELING FOR EEG ANALYSIS: A DIDACTIC EXAMPLE
==============================================================================

SCIENTIFIC BACKGROUND
---------------------
In cognitive neuroscience, we often want to understand how brain activity
relates to behavior. However, we face a fundamental challenge:

    >>> We cannot directly observe the brain state that drives behavior <<<

What we CAN measure:
- EEG signals: electrical activity at the scalp
- Behavior: reaction times, accuracy, choices

The problem:
- EEG is corrupted by measurement noise (sensors, amplifiers, artifacts)
- This noise obscures the true neural dynamics
- Raw EEG may poorly predict behavior due to this noise

THE STATE-SPACE SOLUTION
------------------------
State-space models (like the Kalman filter) address this by:

1. MODELING THE DYNAMICS: We assume the hidden brain state evolves smoothly
   over trials according to known dynamics (e.g., autoregressive process)

2. MODELING THE OBSERVATIONS: We assume EEG is a noisy measurement of this
   hidden state

3. INFERRING THE HIDDEN STATE: The Kalman filter combines our dynamic model
   with the observations to optimally estimate the true state

THE MODEL IN THIS PROJECT
-------------------------
Latent state (hidden, to be inferred):
    x_t = a × x_{t-1} + process_noise
    
    The "true" neural state evolves with some persistence (parameter a)
    and some random variation (process_noise).

EEG observation (what we measure):
    y_t = C × x_t + observation_noise
    
    We observe a noisy version of the true state.

Reaction time (behavioral output):
    RT_t = b - d × x_t + behavioral_noise
    
    Behavior is influenced by the TRUE state, not by noise.
    Higher activation -> faster responses.

KEY INSIGHT
-----------
Because RT is driven by the TRUE latent state (not by observation noise),
a method that can filter out observation noise should predict RT better
than raw EEG.

This is exactly what the Kalman filter does

EXPECTED RESULT
---------------
Correlation(estimated_state, RT) > Correlation(raw_EEG, RT)

This demonstrates the value of state-space modeling in neuroscience.

WHY THIS MATTERS FOR EEG RESEARCH
---------------------------------
1. TRIAL HISTORY EFFECTS: Brain states carry over between trials. The
   autoregressive dynamics capture this persistence.

2. NOISE AND UNCERTAINTY: All neural measurements are noisy. Ignoring this
   leads to underestimating brain-behavior relationships.

3. LATENT VARIABLES: The "true" brain state is a theoretical construct.
   We infer it from multiple noisy measurements.

4. MODELING vs MEASURING: Instead of just measuring (EEG), we model the
   process that generated the measurements.

USAGE
-----
Run this script to execute the full analysis:
    
    python main.py

Or import individual modules:
    
    from simulate_data import simulate_experiment
    from kalman_filter import apply_kalman_filter
    from analyze import compute_correlations
    from plot_results import create_all_figures

DEPENDENCIES
------------
- numpy: numerical computing
- scipy: statistical tests
- matplotlib: visualization
- pykalman (optional): alternative Kalman implementation

==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from simulate_data import simulate_experiment
from kalman_filter import apply_kalman_filter
from analyze import (
    compute_correlations, 
    run_regression_analysis, 
    print_analysis_summary
)
from plot_results import create_all_figures


def main():
    """
    Run the complete state-space modeling demonstration.
    
    This function:
    1. Simulates synthetic EEG data with a known latent state
    2. Applies Kalman filtering to recover the latent state
    3. Analyzes whether the recovered state predicts behavior better than raw EEG
    4. Generates visualizations of the results
    """
    
    print("\n" + "=" * 70)
    print("STATE-SPACE MODELING FOR EEG: DEMONSTRATION")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: SIMULATE DATA
    # =========================================================================
    print("\n[STEP 1] Simulating EEG-like data with latent state...")
    
    # Model parameters (you can modify these to explore different scenarios)
    params = {
        'n_trials': 200,              # Number of trials
        'a': 0.9,                     # State persistence (0-1, higher = more stable)
        'process_noise_std': 1.0,     # State variability between trials
        'C': 1.0,                     # Observation gain
        'observation_noise_std': 2.0, # EEG measurement noise
        'b': 500.0,                   # Baseline RT (ms)
        'd': 20.0,                    # State -> RT coupling strength
        'behavioral_noise_std': 30.0, # RT variability
        'seed': 42                    # For reproducibility
    }
    
    data = simulate_experiment(**params)
    
    print(f"   Generated {params['n_trials']} trials")
    print(f"   Latent state: mean={data['latent_state'].mean():.2f}, "
          f"std={data['latent_state'].std():.2f}")
    print(f"   EEG: mean={data['eeg'].mean():.2f}, std={data['eeg'].std():.2f}")
    print(f"   RT: mean={data['rt'].mean():.1f}ms, std={data['rt'].std():.1f}ms")
    
    # =========================================================================
    # STEP 2: APPLY KALMAN FILTER
    # =========================================================================
    print("\n[STEP 2] Applying Kalman filter to recover latent state from EEG...")
    
    # Note: In real analysis, we wouldn't know the true parameters
    # Here we use the true parameters for demonstration purposes
    kf_results = apply_kalman_filter(
        y=data['eeg'],
        a=params['a'],
        C=params['C'],
        process_noise_var=params['process_noise_std'] ** 2,
        observation_noise_var=params['observation_noise_std'] ** 2
    )
    
    # Compute recovery accuracy
    from scipy.stats import pearsonr
    r_recovery, _ = pearsonr(data['latent_state'], kf_results['smoothed_state'])
    print(f"   State recovery accuracy: r = {r_recovery:.3f}")
    print(f"   (Smoothed Kalman estimate vs true latent state)")
    
    # =========================================================================
    # STEP 3: ANALYZE BEHAVIORAL PREDICTION
    # =========================================================================
    print("\n[STEP 3] Analyzing relationship with reaction time...")
    
    correlations = compute_correlations(
        latent_state_true=data['latent_state'],
        latent_state_estimated=kf_results['smoothed_state'],
        eeg=data['eeg'],
        rt=data['rt']
    )
    
    regression = run_regression_analysis(
        latent_state_estimated=kf_results['smoothed_state'],
        eeg=data['eeg'],
        rt=data['rt']
    )
    
    # Print detailed results
    print_analysis_summary(correlations, regression)
    
    # =========================================================================
    # STEP 4: VISUALIZE RESULTS
    # =========================================================================
    print("[STEP 4] Creating visualizations...")
    
    figures = create_all_figures(
        latent_state_true=data['latent_state'],
        latent_state_estimated=kf_results['smoothed_state'],
        eeg=data['eeg'],
        rt=data['rt']
    )
    
    print("   Created figures:")
    for name in figures.keys():
        print(f"      - {name}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: KEY TAKEAWAYS")
    print("=" * 70)
    
    r_eeg_rt = correlations['with_rt']['raw_eeg']['r']
    r_est_rt = correlations['with_rt']['estimated_state']['r']
    improvement = abs(r_est_rt) - abs(r_eeg_rt)
    
    print(f"""
This demonstration shows that:

1. RAW EEG is noisy:
   - Correlation with true state: r = {correlations['with_true_state']['raw_eeg']['r']:.3f}
   - Some signal, but lots of noise

2. KALMAN FILTER recovers the latent state:
   - Correlation with true state: r = {correlations['with_true_state']['estimated_state']['r']:.3f}
   - Much closer to the ground truth

3. BEHAVIORAL PREDICTION improves with filtering:
   - Raw EEG -> RT: r = {r_eeg_rt:.3f}
   - Estimated state -> RT: r = {r_est_rt:.3f}
   - Improvement: +{improvement:.3f} in |r|

This is why state-space models are valuable for EEG analysis:
They separate true neural dynamics from measurement noise,
revealing stronger brain-behavior relationships.
""")
    print("=" * 70)
    print("\nClose the figure windows to exit.")
    
    plt.show()


if __name__ == "__main__":
    main()
