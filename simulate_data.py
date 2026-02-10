"""
simulate_data.py - Generate synthetic EEG-like data with latent states

CONCEPTUAL OVERVIEW:
====================
In neuroscience, we often assume that behavior and neural signals are driven by
hidden (latent) brain states that we cannot directly observe. This module simulates
such a scenario:

1. LATENT STATE (x_t): A hidden "neural readiness" that evolves over trials
   - Modeled as an AR(1) process: x_t = a * x_{t-1} + noise
   - The parameter 'a' (< 1) controls how much the state persists across trials
   - Represents trial-to-trial fluctuations in attention, arousal, or preparedness

2. EEG OBSERVATION (y_t): What we actually measure
   - y_t = C * x_t + observation_noise
   - The EEG is a noisy reflection of the true state
   - We cannot see x_t directly; we only see this corrupted version

3. REACTION TIME (RT_t): Behavioral output
   - RT_t = b - d * x_t + behavioral_noise
   - Higher "readiness" (higher x_t) leads to faster responses (lower RT)
   - Also corrupted by noise (motor variability, lapses, etc.)

KEY INSIGHT:
The challenge is to recover x_t from y_t, because x_t is what truly drives behavior.
Raw EEG (y_t) is too noisy to predict behavior well.
"""

import numpy as np
from typing import Dict, Tuple


def simulate_latent_state(
    n_trials: int,
    a: float = 0.9,
    process_noise_std: float = 1.0,
    seed: int = None
) -> np.ndarray:
    """
    Simulate the latent neural state as an AR(1) process.
    
    The latent state evolves according to:
        x_t = a * x_{t-1} + process_noise
    
    Parameters
    ----------
    n_trials : int
        Number of trials to simulate
    a : float
        Autoregressive coefficient (0 < a < 1 for stability)
        Higher values = more temporal smoothness (slower changes)
        Typical values: 0.8 - 0.95
    process_noise_std : float
        Standard deviation of the process noise (innovation)
        Controls how much the state can change between trials
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    x : np.ndarray
        Latent state sequence (n_trials,)
    
    Notes
    -----
    The AR(1) model captures "trial history effects" - the idea that
    brain states carry over from one trial to the next. This is why
    reaction times are often autocorrelated in experiments.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize the state array
    x = np.zeros(n_trials)
    
    # Generate all process noise at once
    process_noise = np.random.normal(0, process_noise_std, n_trials)
    
    # First trial: just the noise (no history)
    x[0] = process_noise[0]
    
    # Evolve the state across trials
    for t in range(1, n_trials):
        x[t] = a * x[t - 1] + process_noise[t]
    
    return x


def generate_eeg_observations(
    x: np.ndarray,
    C: float = 1.0,
    observation_noise_std: float = 2.0,
    seed: int = None
) -> np.ndarray:
    """
    Generate noisy EEG observations from the latent state.
    
    The observation model is:
        y_t = C * x_t + observation_noise
    
    Parameters
    ----------
    x : np.ndarray
        True latent state sequence
    C : float
        Observation gain (scaling factor)
        In a real model, this would be a matrix mapping states to sensors
    observation_noise_std : float
        Standard deviation of measurement noise
        Higher values = more noisy EEG (less reliable observations)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    y : np.ndarray
        Observed EEG signal (n_trials,)
    
    Notes
    -----
    The observation noise represents:
    - Sensor noise (electrode impedance, amplifier noise)
    - Biological noise (other neural activity not related to our process)
    - Processing artifacts (muscle, eye movements)
    
    This is why raw EEG is a poor predictor of behavior - too much noise!
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_trials = len(x)
    observation_noise = np.random.normal(0, observation_noise_std, n_trials)
    
    # Linear observation model
    y = C * x + observation_noise
    
    return y


def generate_reaction_times(
    x: np.ndarray,
    b: float = 500.0,
    d: float = 20.0,
    behavioral_noise_std: float = 30.0,
    seed: int = None
) -> np.ndarray:
    """
    Generate reaction times influenced by the latent state.
    
    The behavioral model is:
        RT_t = b - d * x_t + behavioral_noise
    
    Parameters
    ----------
    x : np.ndarray
        True latent state sequence
    b : float
        Baseline reaction time (ms) when x_t = 0
        Typical value: 400-600 ms for simple RT tasks
    d : float
        Coupling strength between state and RT
        Higher d = stronger effect of neural state on behavior
    behavioral_noise_std : float
        Motor/cognitive variability in RT (ms)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    rt : np.ndarray
        Reaction times in milliseconds (n_trials,)
    
    Notes
    -----
    The negative sign (-d * x_t) means:
    - Higher x_t (more "prepared") → lower RT (faster response)
    - Lower x_t (less "prepared") → higher RT (slower response)
    
    This is the relationship we want to recover from the data!
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_trials = len(x)
    behavioral_noise = np.random.normal(0, behavioral_noise_std, n_trials)
    
    # RT decreases when state is high (faster when more prepared)
    rt = b - d * x + behavioral_noise
    
    return rt


def simulate_experiment(
    n_trials: int = 200,
    # State dynamics parameters
    a: float = 0.9,
    process_noise_std: float = 1.0,
    # Observation parameters
    C: float = 1.0,
    observation_noise_std: float = 2.0,
    # Behavioral parameters
    b: float = 500.0,
    d: float = 20.0,
    behavioral_noise_std: float = 30.0,
    # Reproducibility
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Simulate a complete experiment with EEG and behavioral data.
    
    This function generates a synthetic dataset that mimics what we would
    collect in a real experiment, plus the "ground truth" latent state
    that we would never have access to in real data.
    
    Parameters
    ----------
    n_trials : int
        Number of trials (default: 200)
    a : float
        AR coefficient for state dynamics
    process_noise_std : float
        Process noise (state variability)
    C : float
        Observation gain
    observation_noise_std : float
        EEG measurement noise
    b : float
        Baseline RT (ms)
    d : float
        State-to-RT coupling
    behavioral_noise_std : float
        RT variability
    seed : int
        Random seed
    
    Returns
    -------
    data : dict
        Dictionary with:
        - 'latent_state': True hidden state (x_t) - GROUND TRUTH
        - 'eeg': Observed EEG signal (y_t) - WHAT WE MEASURE
        - 'rt': Reaction times (RT_t) - BEHAVIORAL OUTPUT
        - 'params': Model parameters used for simulation
    
    Example
    -------
    >>> data = simulate_experiment(n_trials=200, seed=42)
    >>> print(f"Generated {len(data['eeg'])} trials")
    >>> print(f"Mean RT: {data['rt'].mean():.1f} ms")
    """
    # Set master seed
    np.random.seed(seed)
    
    # Generate latent state
    x = simulate_latent_state(
        n_trials=n_trials,
        a=a,
        process_noise_std=process_noise_std
    )
    
    # Generate EEG observations
    y = generate_eeg_observations(
        x=x,
        C=C,
        observation_noise_std=observation_noise_std
    )
    
    # Generate reaction times
    rt = generate_reaction_times(
        x=x,
        b=b,
        d=d,
        behavioral_noise_std=behavioral_noise_std
    )
    
    # Package everything
    data = {
        'latent_state': x,
        'eeg': y,
        'rt': rt,
        'params': {
            'n_trials': n_trials,
            'a': a,
            'process_noise_std': process_noise_std,
            'C': C,
            'observation_noise_std': observation_noise_std,
            'b': b,
            'd': d,
            'behavioral_noise_std': behavioral_noise_std,
            'seed': seed
        }
    }
    
    return data


# =============================================================================
# Quick test when run directly
# =============================================================================
if __name__ == "__main__":
    # Generate sample data
    data = simulate_experiment(n_trials=200, seed=42)
    
    print("=" * 60)
    print("SIMULATED DATA SUMMARY")
    print("=" * 60)
    print(f"\nNumber of trials: {len(data['eeg'])}")
    print(f"\nLatent state (x_t):")
    print(f"  Mean: {data['latent_state'].mean():.2f}")
    print(f"  Std:  {data['latent_state'].std():.2f}")
    print(f"  Range: [{data['latent_state'].min():.2f}, {data['latent_state'].max():.2f}]")
    print(f"\nEEG observations (y_t):")
    print(f"  Mean: {data['eeg'].mean():.2f}")
    print(f"  Std:  {data['eeg'].std():.2f}")
    print(f"\nReaction times (RT_t):")
    print(f"  Mean: {data['rt'].mean():.1f} ms")
    print(f"  Std:  {data['rt'].std():.1f} ms")
    print(f"  Range: [{data['rt'].min():.1f}, {data['rt'].max():.1f}] ms")
    print("\n" + "=" * 60)
