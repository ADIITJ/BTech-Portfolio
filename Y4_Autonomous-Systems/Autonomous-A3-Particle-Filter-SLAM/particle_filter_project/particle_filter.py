"""
Core particle filter implementation.
Includes RobotSim for ground truth and ParticleFilter for estimation.
"""

import numpy as np
from utils import (wrap_angle, gaussian_pdf, systematic_resample, residual_resample,
                   compute_ess, circular_mean, weighted_covariance)


class RobotSim:
    """
    Simulates true robot motion with noise.
    State: [x, y, theta] where theta is in radians.
    """
    
    def __init__(self, initial_pose, motion_noise_trans, motion_noise_theta):
        """
        initial_pose: [x, y, theta]
        motion_noise_trans: std dev for forward motion (meters)
        motion_noise_theta: std dev for rotation (radians)
        """
        self.pose = np.array(initial_pose, dtype=float)
        self.pose[2] = wrap_angle(self.pose[2])
        self.sigma_trans = motion_noise_trans
        self.sigma_theta = motion_noise_theta
        self.history = [self.pose.copy()]
    
    def move(self, rotation_deg=10.0, forward_dist=1.0):
        """
        Execute motion: rotate left by rotation_deg, then move forward.
        Add Gaussian noise to both rotation and forward distance.
        """
        # Add noise to rotation
        rotation_rad = np.deg2rad(rotation_deg)
        noisy_rotation = rotation_rad + np.random.normal(0, self.sigma_theta)
        
        # Update orientation
        self.pose[2] = wrap_angle(self.pose[2] + noisy_rotation)
        
        # Add noise to forward motion
        noisy_forward = forward_dist + np.random.normal(0, self.sigma_trans)
        
        # Update position
        self.pose[0] += noisy_forward * np.cos(self.pose[2])
        self.pose[1] += noisy_forward * np.sin(self.pose[2])
        
        self.history.append(self.pose.copy())
    
    def measure(self, landmarks, sigma_meas):
        """
        Measure distances to landmarks with Gaussian noise.
        landmarks: Lx2 array of [x, y] positions
        sigma_meas: measurement noise std dev
        Returns: array of noisy distances
        """
        robot_pos = self.pose[:2]
        true_distances = np.linalg.norm(landmarks - robot_pos, axis=1)
        noisy_distances = true_distances + np.random.normal(0, sigma_meas, size=len(landmarks))
        noisy_distances = np.maximum(noisy_distances, 0.0)
        return noisy_distances


class ParticleFilter:
    """
    Particle filter for robot localization.
    Implements predict, update, resample, and estimate steps.
    """
    
    def __init__(self, N, landmarks, motion_noise_trans, motion_noise_theta,
                 meas_noise, resample_method='systematic', rejuvenation_sigma=0.0):
        """
        N: number of particles
        landmarks: Lx2 array of landmark positions
        motion_noise_trans: std dev for particle forward motion
        motion_noise_theta: std dev for particle rotation
        meas_noise: measurement noise std dev
        resample_method: 'systematic' or 'residual'
        rejuvenation_sigma: particle jitter after resampling (0 = disabled)
        """
        self.N = N
        self.landmarks = np.array(landmarks)
        self.sigma_trans = motion_noise_trans
        self.sigma_theta = motion_noise_theta
        self.sigma_meas = meas_noise
        self.resample_method = resample_method
        self.rejuvenation_sigma = rejuvenation_sigma
        self.ess_threshold = N / 3.0
        
        self.particles = None
        self.weights = np.ones(N) / N
        
        self.ess_history = []
        self.weight_failures = 0
        self.low_ess_count = 0
    
    def initialize_particles(self, x_range, y_range, theta_range=None):
        """
        Initialize particles uniformly within specified ranges.
        x_range: (min, max)
        y_range: (min, max)
        theta_range: (min, max) or None for [-pi, pi]
        """
        if theta_range is None:
            theta_range = (-np.pi, np.pi)
        
        self.particles = np.zeros((self.N, 3))
        self.particles[:, 0] = np.random.uniform(x_range[0], x_range[1], self.N)
        self.particles[:, 1] = np.random.uniform(y_range[0], y_range[1], self.N)
        self.particles[:, 2] = np.random.uniform(theta_range[0], theta_range[1], self.N)
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        
        self.weights = np.ones(self.N) / self.N
    
    def predict(self, rotation_deg=10.0, forward_dist=1.0):
        """
        Propagate particles using motion model with noise.
        Each particle samples from motion noise distribution.
        """
        rotation_rad = np.deg2rad(rotation_deg)
        
        # Sample rotation noise for all particles
        noisy_rotation = rotation_rad + np.random.normal(0, self.sigma_theta, self.N)
        
        # Update orientation
        self.particles[:, 2] = wrap_angle(self.particles[:, 2] + noisy_rotation)
        
        # Sample forward motion noise
        noisy_forward = forward_dist + np.random.normal(0, self.sigma_trans, self.N)
        
        # Update positions (vectorized)
        self.particles[:, 0] += noisy_forward * np.cos(self.particles[:, 2])
        self.particles[:, 1] += noisy_forward * np.sin(self.particles[:, 2])
    
    def update(self, measurements):
        """
        Update particle weights based on measurement likelihood.
        Uses log-likelihood for numerical stability.
        measurements: array of distances to landmarks
        """
        # Compute predicted measurements for all particles
        # Shape: (N, L) where N=particles, L=landmarks
        particle_positions = self.particles[:, :2]  # Nx2
        predicted_dists = np.linalg.norm(
            particle_positions[:, np.newaxis, :] - self.landmarks[np.newaxis, :, :],
            axis=2
        )  # NxL
        
        # Compute log-likelihood for each particle
        # p(z|x) = product over landmarks of Gaussian PDF
        # log p(z|x) = sum of log Gaussian PDFs
        log_weights = np.zeros(self.N)
        for i in range(len(self.landmarks)):
            diff = measurements[i] - predicted_dists[:, i]
            log_weights += -0.5 * (diff / self.sigma_meas) ** 2
        
        # Normalize weights using log-sum-exp trick
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        weight_sum = np.sum(weights)
        
        # Handle failure case: all weights near zero
        if weight_sum < 1e-50 or np.isnan(weight_sum):
            self.weights = np.ones(self.N) / self.N
            self.weight_failures += 1
        else:
            self.weights = weights / weight_sum
    
    def resample(self):
        """
        Resample particles based on weights.
        After resampling, reset weights to uniform and optionally add jitter.
        """
        ess = compute_ess(self.weights)
        self.ess_history.append(ess)
        
        if ess < self.ess_threshold:
            self.low_ess_count += 1
        
        if self.resample_method == 'systematic':
            indices = systematic_resample(self.weights, self.N)
        elif self.resample_method == 'residual':
            indices = residual_resample(self.weights, self.N)
        else:
            raise ValueError(f"Unknown resample method: {self.resample_method}")
        
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N
        
        if self.rejuvenation_sigma > 0:
            self.particles[:, 0] += np.random.normal(0, self.rejuvenation_sigma, self.N)
            self.particles[:, 1] += np.random.normal(0, self.rejuvenation_sigma, self.N)
    
    def estimate(self):
        """
        Compute weighted mean estimate of robot pose.
        Returns: [x, y, theta] estimate and 2x2 covariance matrix for x,y
        """
        # Weighted mean for x, y
        mean_x = np.sum(self.weights * self.particles[:, 0])
        mean_y = np.sum(self.weights * self.particles[:, 1])
        
        # Circular mean for theta
        mean_theta = circular_mean(self.particles[:, 2], self.weights)
        
        # Compute covariance
        cov = weighted_covariance(self.particles, self.weights)
        
        return np.array([mean_x, mean_y, mean_theta]), cov
    
    def get_max_weight_particle(self):
        """
        Return the particle with highest weight.
        Returns: particle [x, y, theta] and its index
        """
        idx = np.argmax(self.weights)
        return self.particles[idx].copy(), idx
    
    def get_variance(self, cov):
        """
        Compute trace of covariance matrix (total variance).
        """
        return np.trace(cov)
