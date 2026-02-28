"""
Visualization utilities for particle filter.
Includes per-timestep plots and summary plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from utils import cov_ellipse, wrap_angle


def plot_timestep(particles, weights, true_pose, estimated_pose, cov, landmarks,
                  max_weight_particle, timestep, output_path):
    """
    Create and save a plot for a single timestep.
    Shows particles, true robot, estimate, landmarks, and covariance ellipse.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot particles (size proportional to weight for visibility)
    sizes = 50 * weights / np.max(weights) + 1
    ax.scatter(particles[:, 0], particles[:, 1], s=sizes, c='blue', alpha=0.3, label='Particles')
    
    # Plot max weight particle
    ax.scatter(max_weight_particle[0], max_weight_particle[1], s=200, c='cyan',
               marker='o', edgecolors='black', linewidths=2, label='Max Weight Particle', zorder=5)
    
    # Plot true robot position
    ax.scatter(true_pose[0], true_pose[1], s=300, c='red', marker='*',
               edgecolors='black', linewidths=2, label='True Position', zorder=5)
    
    # Plot estimated position
    ax.scatter(estimated_pose[0], estimated_pose[1], s=200, c='green', marker='s',
               edgecolors='black', linewidths=2, label='Estimate', zorder=5)
    
    # Plot landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=150, c='orange', marker='x',
               linewidths=3, label='Landmarks', zorder=5)
    
    # Plot covariance ellipse
    try:
        ellipse_pts = cov_ellipse(cov, q=0.95)
        ellipse_pts[:, 0] += estimated_pose[0]
        ellipse_pts[:, 1] += estimated_pose[1]
        ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], 'g--', linewidth=2, label='95% Confidence')
    except:
        pass  # Skip if covariance is singular
    
    all_x = np.concatenate([particles[:, 0], landmarks[:, 0], [true_pose[0], estimated_pose[0]]])
    all_y = np.concatenate([particles[:, 1], landmarks[:, 1], [true_pose[1], estimated_pose[1]]])
    margin = 2.0
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Particle Filter - Timestep {timestep}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def plot_rmse_over_time(rmse_values, output_path):
    """
    Plot RMSE vs time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rmse_values, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('RMSE (m)', fontsize=12)
    ax.set_title('Position RMSE over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ess_over_time(ess_values, output_path):
    """
    Plot Effective Sample Size vs time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ess_values, 'r-', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Effective Sample Size', fontsize=12)
    ax.set_title('ESS over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_variance_over_time(variance_values, output_path):
    """
    Plot particle variance (trace of covariance) vs time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(variance_values, 'g-', linewidth=2, marker='^', markersize=4)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Variance (m²)', fontsize=12)
    ax.set_title('Particle Variance over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_heading_error_over_time(heading_errors, output_path):
    """
    Plot heading error vs time.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.rad2deg(heading_errors), 'm-', linewidth=2, marker='d', markersize=4)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Heading Error (degrees)', fontsize=12)
    ax.set_title('Heading Error over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trajectory(true_trajectory, estimated_trajectory, landmarks, output_path):
    """
    Plot complete trajectory comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    true_traj = np.array(true_trajectory)
    est_traj = np.array(estimated_trajectory)
    
    ax.plot(true_traj[:, 0], true_traj[:, 1], 'r-', linewidth=2, marker='o',
            markersize=6, label='True Path', alpha=0.7)
    ax.plot(est_traj[:, 0], est_traj[:, 1], 'g--', linewidth=2, marker='s',
            markersize=6, label='Estimated Path', alpha=0.7)
    
    # Mark start and end
    ax.scatter(true_traj[0, 0], true_traj[0, 1], s=300, c='red', marker='*',
               edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax.scatter(true_traj[-1, 0], true_traj[-1, 1], s=300, c='darkred', marker='X',
               edgecolors='black', linewidths=2, label='End', zorder=5)
    
    # Plot landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=150, c='orange', marker='x',
               linewidths=3, label='Landmarks', zorder=5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Complete Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_summary_plots(stats_list, output_dir):
    """
    Create a grid of summary plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    timesteps = range(len(stats_list))
    rmse_values = [s['rmse'] for s in stats_list]
    ess_values = [s['ess'] for s in stats_list]
    variance_values = [s['variance'] for s in stats_list]
    heading_errors = [s['heading_error'] for s in stats_list]
    
    # RMSE
    axes[0, 0].plot(timesteps, rmse_values, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Timestep', fontsize=11)
    axes[0, 0].set_ylabel('RMSE (m)', fontsize=11)
    axes[0, 0].set_title('Position RMSE', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # ESS
    axes[0, 1].plot(timesteps, ess_values, 'r-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Timestep', fontsize=11)
    axes[0, 1].set_ylabel('ESS', fontsize=11)
    axes[0, 1].set_title('Effective Sample Size', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Variance
    axes[1, 0].plot(timesteps, variance_values, 'g-', linewidth=2, marker='^', markersize=4)
    axes[1, 0].set_xlabel('Timestep', fontsize=11)
    axes[1, 0].set_ylabel('Variance (m²)', fontsize=11)
    axes[1, 0].set_title('Particle Variance', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Heading Error
    axes[1, 1].plot(timesteps, np.rad2deg(heading_errors), 'm-', linewidth=2, marker='d', markersize=4)
    axes[1, 1].set_xlabel('Timestep', fontsize=11)
    axes[1, 1].set_ylabel('Error (degrees)', fontsize=11)
    axes[1, 1].set_title('Heading Error', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/particles_summary.png', dpi=150)
    plt.close()


def plot_comparison_across_experiments(experiment_results, metric_name, output_path):
    """
    Plot a specific metric across multiple experiments.
    experiment_results: list of dicts with 'name' and 'stats' keys
    metric_name: key in stats dict (e.g., 'rmse', 'ess', 'variance')
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for exp in experiment_results:
        values = [s[metric_name] for s in exp['stats']]
        ax.plot(values, linewidth=2, marker='o', markersize=3, label=exp['name'], alpha=0.8)
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.set_title(f'{metric_name.upper()} Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
