"""
plot_results.py - Visualization of state-space modeling results

CONCEPTUAL OVERVIEW:
====================
Visualizations help build intuition about what state-space models do.

This module creates figures that show:
1. How well we recovered the latent state from noisy observations
2. The relationship between neural activity and behavior
3. Why filtering matters for brain-behavior relationships

Good figures tell the story of the model!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


def setup_plot_style():
    """Set up a clean, publication-ready plot style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.dpi': 100
    })


def plot_state_recovery(
    latent_state_true: np.ndarray,
    latent_state_estimated: np.ndarray,
    eeg: np.ndarray,
    trial_range: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comparison of true latent state, estimated state, and raw EEG.
    
    This is THE key figure for understanding state recovery.
    
    Parameters
    ----------
    latent_state_true : np.ndarray
        Ground truth latent state
    latent_state_estimated : np.ndarray
        Kalman-filtered estimate
    eeg : np.ndarray
        Raw EEG observations
    trial_range : tuple, optional
        (start, end) trials to plot (default: first 100)
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib Figure
    """
    setup_plot_style()
    
    if trial_range is None:
        trial_range = (0, min(100, len(latent_state_true)))
    
    start, end = trial_range
    trials = np.arange(start, end)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # ------------------------------------
    # Panel A: True vs Estimated State
    # ------------------------------------
    ax = axes[0]
    ax.plot(trials, latent_state_true[start:end], 
            'b-', linewidth=2, label='True Latent State', alpha=0.8)
    ax.plot(trials, latent_state_estimated[start:end], 
            'r--', linewidth=2, label='Estimated State (Kalman)', alpha=0.8)
    ax.set_ylabel('Latent State\n(arbitrary units)')
    ax.set_title('A. State Recovery: Kalman Filter Recovers Hidden Neural State')
    ax.legend(loc='upper right')
    
    # Add correlation annotation
    from scipy.stats import pearsonr
    r, _ = pearsonr(latent_state_true, latent_state_estimated)
    ax.text(0.02, 0.95, f'r = {r:.3f}', transform=ax.transAxes, 
            fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ------------------------------------
    # Panel B: Raw EEG (noisy observation)
    # ------------------------------------
    ax = axes[1]
    ax.plot(trials, eeg[start:end], 
            'gray', linewidth=1, alpha=0.7, label='Raw EEG (noisy)')
    ax.plot(trials, latent_state_true[start:end], 
            'b-', linewidth=2, alpha=0.5, label='True State (hidden)')
    ax.set_ylabel('EEG Signal\n(arbitrary units)')
    ax.set_title('B. The Problem: Raw EEG is a Noisy Version of the True State')
    ax.legend(loc='upper right')
    
    # ------------------------------------
    # Panel C: Zoom on a few trials
    # ------------------------------------
    ax = axes[2]
    zoom_start = start
    zoom_end = min(start + 30, end)
    zoom_trials = np.arange(zoom_start, zoom_end)
    
    ax.plot(zoom_trials, latent_state_true[zoom_start:zoom_end], 
            'b-o', linewidth=2, markersize=4, label='True State')
    ax.plot(zoom_trials, latent_state_estimated[zoom_start:zoom_end], 
            'r--s', linewidth=2, markersize=4, label='Estimated')
    ax.plot(zoom_trials, eeg[zoom_start:zoom_end], 
            'gray', marker='^', linewidth=1, markersize=3, alpha=0.5, label='Raw EEG')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Signal Value')
    ax.set_title('C. Zoomed View: Filtering Removes Observation Noise')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_behavioral_relationships(
    latent_state_true: np.ndarray,
    latent_state_estimated: np.ndarray,
    eeg: np.ndarray,
    rt: np.ndarray,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot relationships between neural measures and reaction time.
    
    This demonstrates the key finding: filtered state predicts RT better.
    
    Parameters
    ----------
    latent_state_true : np.ndarray
        Ground truth latent state
    latent_state_estimated : np.ndarray
        Kalman-filtered estimate
    eeg : np.ndarray
        Raw EEG observations
    rt : np.ndarray
        Reaction times
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib Figure
    """
    setup_plot_style()
    
    from scipy.stats import pearsonr
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Color scheme
    colors = {
        'true': '#2196F3',      # Blue
        'estimated': '#E91E63', # Pink/Red
        'eeg': '#607D8B'        # Gray
    }
    
    # ------------------------------------
    # Row 1: Scatter plots with RT
    # ------------------------------------
    
    # True State vs RT
    ax = axes[0, 0]
    r, p = pearsonr(latent_state_true, rt)
    ax.scatter(latent_state_true, rt, alpha=0.5, c=colors['true'], s=20)
    # Add regression line
    z = np.polyfit(latent_state_true, rt, 1)
    x_line = np.linspace(latent_state_true.min(), latent_state_true.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k-', linewidth=2)
    ax.set_xlabel('True Latent State')
    ax.set_ylabel('Reaction Time (ms)')
    ax.set_title(f'True State vs RT\nr = {r:.3f}')
    
    # Estimated State vs RT
    ax = axes[0, 1]
    r, p = pearsonr(latent_state_estimated, rt)
    ax.scatter(latent_state_estimated, rt, alpha=0.5, c=colors['estimated'], s=20)
    z = np.polyfit(latent_state_estimated, rt, 1)
    x_line = np.linspace(latent_state_estimated.min(), latent_state_estimated.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k-', linewidth=2)
    ax.set_xlabel('Estimated Latent State')
    ax.set_ylabel('Reaction Time (ms)')
    ax.set_title(f'Estimated State vs RT\nr = {r:.3f}')
    
    # Raw EEG vs RT
    ax = axes[0, 2]
    r, p = pearsonr(eeg, rt)
    ax.scatter(eeg, rt, alpha=0.5, c=colors['eeg'], s=20)
    z = np.polyfit(eeg, rt, 1)
    x_line = np.linspace(eeg.min(), eeg.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k-', linewidth=2)
    ax.set_xlabel('Raw EEG')
    ax.set_ylabel('Reaction Time (ms)')
    ax.set_title(f'Raw EEG vs RT\nr = {r:.3f}')
    
    # ------------------------------------
    # Row 2: Summary and comparisons
    # ------------------------------------
    
    # Bar chart of correlations
    ax = axes[1, 0]
    r_true, _ = pearsonr(latent_state_true, rt)
    r_est, _ = pearsonr(latent_state_estimated, rt)
    r_eeg, _ = pearsonr(eeg, rt)
    
    labels = ['True State\n(ground truth)', 'Estimated State\n(Kalman)', 'Raw EEG']
    correlations = [abs(r_true), abs(r_est), abs(r_eeg)]
    bar_colors = [colors['true'], colors['estimated'], colors['eeg']]
    
    bars = ax.bar(labels, correlations, color=bar_colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('|Correlation| with RT')
    ax.set_title('Predictive Power Comparison')
    ax.set_ylim(0, max(correlations) * 1.2)
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
    
    # State recovery scatter
    ax = axes[1, 1]
    r, _ = pearsonr(latent_state_true, latent_state_estimated)
    ax.scatter(latent_state_true, latent_state_estimated, 
               alpha=0.5, c=colors['estimated'], s=20)
    # Perfect recovery line
    lims = [min(latent_state_true.min(), latent_state_estimated.min()),
            max(latent_state_true.max(), latent_state_estimated.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, label='Perfect recovery')
    ax.set_xlabel('True Latent State')
    ax.set_ylabel('Estimated Latent State')
    ax.set_title(f'State Recovery Accuracy\nr = {r:.3f}')
    ax.legend()
    
    # Improvement visualization
    ax = axes[1, 2]
    improvement = abs(r_est) - abs(r_eeg)
    
    ax.bar(['Raw EEG', 'Kalman\n(improvement)'], 
           [abs(r_eeg), abs(r_est) - abs(r_eeg)],
           color=[colors['eeg'], colors['estimated']],
           edgecolor='black', linewidth=1,
           bottom=[0, abs(r_eeg)])
    
    ax.axhline(y=abs(r_true), color=colors['true'], linestyle='--', 
               linewidth=2, label=f'True state ceiling: {abs(r_true):.3f}')
    ax.set_ylabel('|Correlation| with RT')
    ax.set_title(f'Improvement from Filtering\nΔ|r| = {improvement:+.3f}')
    ax.legend(loc='lower right')
    ax.set_ylim(0, max(correlations) * 1.2)
    
    plt.tight_layout()
    return fig


def plot_time_series_with_behavior(
    latent_state_estimated: np.ndarray,
    rt: np.ndarray,
    trial_range: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot estimated state and RT over trials to show covariation.
    
    Parameters
    ----------
    latent_state_estimated : np.ndarray
        Kalman-filtered state
    rt : np.ndarray
        Reaction times
    trial_range : tuple, optional
        (start, end) trials to plot
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib Figure
    """
    setup_plot_style()
    
    if trial_range is None:
        trial_range = (0, min(100, len(rt)))
    
    start, end = trial_range
    trials = np.arange(start, end)
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot state on left axis
    color1 = '#E91E63'
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Estimated Latent State', color=color1)
    line1, = ax1.plot(trials, latent_state_estimated[start:end], 
                      color=color1, linewidth=2, label='Estimated State')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot RT on right axis
    ax2 = ax1.twinx()
    color2 = '#2196F3'
    ax2.set_ylabel('Reaction Time (ms)', color=color2)
    line2, = ax2.plot(trials, rt[start:end], 
                      color=color2, linewidth=2, alpha=0.7, label='Reaction Time')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.invert_yaxis()  # Invert so high state = low RT visually align
    
    # Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    ax1.set_title('Neural State and Behavior Covary Over Trials\n'
                  '(RT axis inverted: high state → fast response)')
    
    plt.tight_layout()
    return fig


def create_all_figures(
    latent_state_true: np.ndarray,
    latent_state_estimated: np.ndarray,
    eeg: np.ndarray,
    rt: np.ndarray,
    save_path: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Generate all figures for the analysis.
    
    Parameters
    ----------
    latent_state_true : np.ndarray
        Ground truth latent state
    latent_state_estimated : np.ndarray
        Kalman-filtered estimate
    eeg : np.ndarray
        Raw EEG observations
    rt : np.ndarray
        Reaction times
    save_path : str, optional
        Base path to save figures (without extension)
    
    Returns
    -------
    figures : dict
        Dictionary of matplotlib Figure objects
    """
    figures = {}
    
    # Figure 1: State recovery
    figures['state_recovery'] = plot_state_recovery(
        latent_state_true, latent_state_estimated, eeg
    )
    
    # Figure 2: Behavioral relationships
    figures['behavioral'] = plot_behavioral_relationships(
        latent_state_true, latent_state_estimated, eeg, rt
    )
    
    # Figure 3: Time series
    figures['timeseries'] = plot_time_series_with_behavior(
        latent_state_estimated, rt
    )
    
    # Save if path provided
    if save_path:
        for name, fig in figures.items():
            fig.savefig(f"{save_path}_{name}.png", dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}_{name}.png")
    
    return figures


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
    
    # Create figures
    figures = create_all_figures(
        latent_state_true=data['latent_state'],
        latent_state_estimated=kf_results['smoothed_state'],
        eeg=data['eeg'],
        rt=data['rt']
    )
    
    print("\nFigures created successfully!")
    print("Close the plot window to exit.")
    plt.show()
