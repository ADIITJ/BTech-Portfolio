"""
Main script to run particle filter experiments.
Runs multiple experiments varying particle counts and noise levels.
Produces plots, JSON results, CSV stats, and optional animations.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from particle_filter import RobotSim, ParticleFilter
from visualize import (plot_timestep, plot_rmse_over_time, plot_ess_over_time,
                       plot_variance_over_time, plot_heading_error_over_time,
                       plot_trajectory, create_summary_plots,
                       plot_comparison_across_experiments)
from utils import wrap_angle


def run_single_experiment(config, save_frames=True):
    """
    Run a single particle filter experiment with given configuration.
    Returns statistics and paths to outputs.
    """
    # Set random seed for reproducibility
    np.random.seed(config['seed'])
    
    # Extract parameters
    N = config['num_particles']
    T = config['num_timesteps']
    landmarks = np.array(config['landmarks'])
    
    # Noise parameters
    robot_motion_trans = config['robot_motion_noise_trans']
    robot_motion_theta = config['robot_motion_noise_theta']
    particle_motion_trans = config['particle_motion_noise_trans']
    particle_motion_theta = config['particle_motion_noise_theta']
    meas_noise = config['meas_noise']
    
    # Create output directory
    exp_name = config['name']
    exp_dir = f"experiments/{exp_name}"
    frames_dir = f"{exp_dir}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize robot and particle filter
    initial_pose = config['initial_pose']
    robot = RobotSim(initial_pose, robot_motion_trans, robot_motion_theta)
    
    pf = ParticleFilter(
        N=N,
        landmarks=landmarks,
        motion_noise_trans=particle_motion_trans,
        motion_noise_theta=particle_motion_theta,
        meas_noise=meas_noise,
        resample_method=config.get('resample_method', 'systematic'),
        rejuvenation_sigma=config.get('rejuvenation_sigma', 0.0)
    )
    
    init_range = config.get('init_range', 5.0)
    pf.initialize_particles(
        x_range=(initial_pose[0] - init_range, initial_pose[0] + init_range),
        y_range=(initial_pose[1] - init_range, initial_pose[1] + init_range),
        theta_range=(-np.pi, np.pi)
    )
    
    # Storage for statistics
    stats_list = []
    true_trajectory = [robot.pose.copy()]
    estimated_trajectory = []
    
    # Run simulation
    for t in tqdm(range(T), desc=f"Running {exp_name}", leave=False):
        # Predict
        pf.predict(rotation_deg=10.0, forward_dist=1.0)
        
        # Robot moves
        robot.move(rotation_deg=10.0, forward_dist=1.0)
        
        # Measure
        measurements = robot.measure(landmarks, meas_noise)
        
        # Update
        pf.update(measurements)
        
        # Resample
        pf.resample()
        
        # Estimate
        estimated_pose, cov = pf.estimate()
        estimated_trajectory.append(estimated_pose.copy())
        
        # Compute metrics
        true_pose = robot.pose
        rmse = np.sqrt((estimated_pose[0] - true_pose[0])**2 + 
                      (estimated_pose[1] - true_pose[1])**2)
        heading_error = wrap_angle(estimated_pose[2] - true_pose[2])
        variance = pf.get_variance(cov)
        ess = pf.ess_history[-1] if pf.ess_history else N
        max_particle, max_idx = pf.get_max_weight_particle()
        
        # Store statistics
        stats = {
            'timestep': t,
            'true_x': float(true_pose[0]),
            'true_y': float(true_pose[1]),
            'true_theta': float(true_pose[2]),
            'est_x': float(estimated_pose[0]),
            'est_y': float(estimated_pose[1]),
            'est_theta': float(estimated_pose[2]),
            'rmse': float(rmse),
            'heading_error': float(heading_error),
            'variance': float(variance),
            'ess': float(ess),
            'max_weight_particle_idx': int(max_idx),
            'max_weight_particle_x': float(max_particle[0]),
            'max_weight_particle_y': float(max_particle[1])
        }
        stats_list.append(stats)
        
        # Save frame
        if save_frames:
            frame_path = f"{frames_dir}/frame_{t:03d}.png"
            plot_timestep(
                pf.particles, pf.weights, true_pose, estimated_pose,
                cov, landmarks, max_particle, t, frame_path
            )
    
    # Save results JSON
    results = {
        'config': config,
        'stats': stats_list,
        'final_rmse': float(stats_list[-1]['rmse']),
        'mean_rmse': float(np.mean([s['rmse'] for s in stats_list])),
        'mean_ess': float(np.mean([s['ess'] for s in stats_list])),
        'weight_failures': int(pf.weight_failures),
        'diverged': bool(stats_list[-1]['rmse'] > config.get('divergence_threshold', 10.0))
    }
    
    with open(f"{exp_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save stats CSV
    df = pd.DataFrame(stats_list)
    df.to_csv(f"{exp_dir}/stats.csv", index=False)
    
    # Create summary plots
    plot_rmse_over_time([s['rmse'] for s in stats_list], f"{exp_dir}/rmse_plot.png")
    plot_ess_over_time([s['ess'] for s in stats_list], f"{exp_dir}/ess_plot.png")
    plot_variance_over_time([s['variance'] for s in stats_list], f"{exp_dir}/variance_plot.png")
    plot_heading_error_over_time([s['heading_error'] for s in stats_list], f"{exp_dir}/heading_error_plot.png")
    plot_trajectory(true_trajectory, estimated_trajectory, landmarks, f"{exp_dir}/trajectory_plot.png")
    create_summary_plots(stats_list, exp_dir)
    
    # Create animation if requested
    if config.get('create_animation', False):
        try:
            import imageio
            frames = []
            for t in range(T):
                frame_path = f"{frames_dir}/frame_{t:03d}.png"
                if os.path.exists(frame_path):
                    frames.append(imageio.imread(frame_path))
            if frames:
                imageio.mimsave(f"{exp_dir}/animation.gif", frames, duration=0.5)
        except ImportError:
            print(f"  Warning: imageio not available, skipping animation for {exp_name}")
    
    return results


def create_experiment_configs(quick_mode=False):
    """
    Generate experiment configurations.
    Varies particle count, motion noise, and measurement noise.
    """
    configs = []
    
    # Default landmarks
    landmarks = [
        [5, 5],
        [10, 10],
        [5, 15],
        [15, 5],
        [15, 15],
        [10, 20]
    ]
    
    # Base configuration
    base_config = {
        'num_timesteps': 8 if quick_mode else 30,
        'initial_pose': [0.0, 0.0, np.pi/4],
        'landmarks': landmarks,
        'seed': 0,
        'init_range': 5.0,
        'resample_method': 'systematic',
        'rejuvenation_sigma': 0.0,
        'create_animation': False,
        'divergence_threshold': 10.0
    }
    
    if quick_mode:
        # Quick test configuration
        config = base_config.copy()
        config.update({
            'name': 'quick_test',
            'num_particles': 200,
            'robot_motion_noise_trans': 0.02,
            'robot_motion_noise_theta': np.deg2rad(1),
            'particle_motion_noise_trans': 0.02,
            'particle_motion_noise_theta': np.deg2rad(1),
            'meas_noise': 0.1
        })
        configs.append(config)
    else:
        # Full experiment suite
        particle_counts = [100, 300, 1000]
        motion_noise_configs = [
            {'trans': 0.02, 'theta': np.deg2rad(1)},
            {'trans': 0.1, 'theta': np.deg2rad(5)}
        ]
        meas_noise_values = [0.05, 0.2, 0.5]
        
        exp_id = 1
        
        # Vary particle count with low noise
        for N in particle_counts:
            config = base_config.copy()
            config.update({
                'name': f'exp_{exp_id:02d}_N{N}_low_noise',
                'num_particles': N,
                'robot_motion_noise_trans': 0.02,
                'robot_motion_noise_theta': np.deg2rad(1),
                'particle_motion_noise_trans': 0.02,
                'particle_motion_noise_theta': np.deg2rad(1),
                'meas_noise': 0.05
            })
            configs.append(config)
            exp_id += 1
        
        # Vary motion noise with fixed particle count
        for motion_noise in motion_noise_configs:
            config = base_config.copy()
            config.update({
                'name': f'exp_{exp_id:02d}_N300_trans{motion_noise["trans"]:.2f}_theta{np.rad2deg(motion_noise["theta"]):.0f}deg',
                'num_particles': 300,
                'robot_motion_noise_trans': motion_noise['trans'],
                'robot_motion_noise_theta': motion_noise['theta'],
                'particle_motion_noise_trans': motion_noise['trans'],
                'particle_motion_noise_theta': motion_noise['theta'],
                'meas_noise': 0.1
            })
            configs.append(config)
            exp_id += 1
        
        # Vary measurement noise with fixed particle count
        for meas_noise in meas_noise_values:
            config = base_config.copy()
            config.update({
                'name': f'exp_{exp_id:02d}_N300_meas{meas_noise:.2f}',
                'num_particles': 300,
                'robot_motion_noise_trans': 0.02,
                'robot_motion_noise_theta': np.deg2rad(1),
                'particle_motion_noise_trans': 0.02,
                'particle_motion_noise_theta': np.deg2rad(1),
                'meas_noise': meas_noise
            })
            configs.append(config)
            exp_id += 1
        
        # High noise challenge
        config = base_config.copy()
        config.update({
            'name': f'exp_{exp_id:02d}_N1000_high_noise_challenge',
            'num_particles': 1000,
            'robot_motion_noise_trans': 0.1,
            'robot_motion_noise_theta': np.deg2rad(5),
            'particle_motion_noise_trans': 0.1,
            'particle_motion_noise_theta': np.deg2rad(5),
            'meas_noise': 0.5,
            'rejuvenation_sigma': 0.05
        })
        configs.append(config)
        exp_id += 1
        
        # Residual resampling test
        config = base_config.copy()
        config.update({
            'name': f'exp_{exp_id:02d}_N300_residual_resample',
            'num_particles': 300,
            'robot_motion_noise_trans': 0.02,
            'robot_motion_noise_theta': np.deg2rad(1),
            'particle_motion_noise_trans': 0.02,
            'particle_motion_noise_theta': np.deg2rad(1),
            'meas_noise': 0.1,
            'resample_method': 'residual'
        })
        configs.append(config)
    
    return configs


def generate_summary_report(all_results):
    """
    Generate summary CSV and comparison plots across all experiments.
    """
    # Create summary dataframe
    summary_data = []
    for result in all_results:
        config = result['config']
        summary_data.append({
            'experiment': config['name'],
            'num_particles': config['num_particles'],
            'motion_noise_trans': config['robot_motion_noise_trans'],
            'motion_noise_theta_deg': np.rad2deg(config['robot_motion_noise_theta']),
            'meas_noise': config['meas_noise'],
            'resample_method': config['resample_method'],
            'rejuvenation_sigma': config['rejuvenation_sigma'],
            'final_rmse': result['final_rmse'],
            'mean_rmse': result['mean_rmse'],
            'mean_ess': result['mean_ess'],
            'weight_failures': result['weight_failures'],
            'diverged': result['diverged']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv('experiments/summary.csv', index=False)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Create comparison plots
    exp_results_for_plots = [
        {'name': r['config']['name'], 'stats': r['stats']}
        for r in all_results
    ]
    
    plot_comparison_across_experiments(
        exp_results_for_plots, 'rmse',
        'experiments/comparison_rmse.png'
    )
    plot_comparison_across_experiments(
        exp_results_for_plots, 'ess',
        'experiments/comparison_ess.png'
    )
    plot_comparison_across_experiments(
        exp_results_for_plots, 'variance',
        'experiments/comparison_variance.png'
    )


def main():
    parser = argparse.ArgumentParser(description='Run particle filter experiments')
    parser.add_argument('--quick', action='store_true',
                       help='Run a quick test with reduced timesteps')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate HTML summary report')
    args = parser.parse_args()
    
    print("="*80)
    print("PARTICLE FILTER SIMULATION - EXPERIMENT SUITE")
    print("="*80)
    
    # Generate experiment configurations
    configs = create_experiment_configs(quick_mode=args.quick)
    
    print(f"\nRunning {len(configs)} experiment(s)...\n")
    
    # Run all experiments
    all_results = []
    for config in configs:
        print(f"\nStarting: {config['name']}")
        result = run_single_experiment(config, save_frames=True)
        all_results.append(result)
        print(f"  Final RMSE: {result['final_rmse']:.4f} m")
        print(f"  Mean RMSE: {result['mean_rmse']:.4f} m")
        print(f"  Mean ESS: {result['mean_ess']:.2f}")
    
    # Generate summary report
    if not args.quick:
        generate_summary_report(all_results)
    
    # Generate HTML report if requested
    if args.generate_report:
        generate_html_report(all_results)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: experiments/")
    print("="*80 + "\n")


def generate_html_report(all_results):
    """
    Generate a simple HTML summary report.
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Particle Filter Experiment Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .diverged { background-color: #ffcccc; }
            img { max-width: 100%; height: auto; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Particle Filter Experiment Report</h1>
        <p>Generated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        
        <h2>Summary Table</h2>
        <table>
            <tr>
                <th>Experiment</th>
                <th>Particles</th>
                <th>Motion Noise (m)</th>
                <th>Meas Noise (m)</th>
                <th>Final RMSE (m)</th>
                <th>Mean ESS</th>
                <th>Status</th>
            </tr>
    """
    
    for result in all_results:
        config = result['config']
        row_class = 'diverged' if result['diverged'] else ''
        status = 'DIVERGED' if result['diverged'] else 'OK'
        
        html += f"""
            <tr class='{row_class}'>
                <td>{config['name']}</td>
                <td>{config['num_particles']}</td>
                <td>{config['robot_motion_noise_trans']:.3f}</td>
                <td>{config['meas_noise']:.3f}</td>
                <td>{result['final_rmse']:.4f}</td>
                <td>{result['mean_ess']:.2f}</td>
                <td>{status}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Comparison Plots</h2>
        <h3>RMSE Comparison</h3>
        <img src="comparison_rmse.png" alt="RMSE Comparison">
        
        <h3>ESS Comparison</h3>
        <img src="comparison_ess.png" alt="ESS Comparison">
        
        <h3>Variance Comparison</h3>
        <img src="comparison_variance.png" alt="Variance Comparison">
        
    </body>
    </html>
    """
    
    with open('experiments/report.html', 'w') as f:
        f.write(html)
    
    print("\nHTML report generated: experiments/report.html")


if __name__ == '__main__':
    main()
