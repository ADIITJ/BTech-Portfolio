"""
Utility functions for particle filter implementation.
Includes angle normalization, Gaussian PDF, resampling, and covariance ellipse.
"""

import numpy as np
from scipy import stats


def wrap_angle(angle):
    """
    Normalize angle to [-pi, pi].
    Works for scalars and arrays.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def gaussian_pdf(x, mu, sigma):
    """
    Compute Gaussian PDF at x given mean mu and std sigma.
    Returns probability density.
    """
    return stats.norm.pdf(x, loc=mu, scale=sigma)


def systematic_resample(weights, N):
    """
    Systematic resampling: deterministic low-variance resampling.
    Returns array of N indices sampled from distribution given by weights.
    """
    weights = weights / np.sum(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    
    positions = (np.arange(N) + np.random.uniform()) / N
    indices = np.searchsorted(cumsum, positions)
    indices = np.minimum(indices, N - 1)
    
    return indices


def residual_resample(weights, N):
    """
    Residual resampling: deterministic for integer parts, stochastic for fractional.
    Returns array of N indices.
    """
    weights = weights / np.sum(weights)
    num_copies = (weights * N).astype(int)
    residual = weights * N - num_copies
    
    indices = []
    for i, n in enumerate(num_copies):
        indices.extend([i] * n)
    
    k = N - len(indices)
    if k > 0:
        residual_sum = np.sum(residual)
        if residual_sum > 1e-10:
            residual = residual / residual_sum
            residual_indices = np.random.choice(len(weights), size=k, p=residual, replace=True)
            indices.extend(residual_indices)
        else:
            indices.extend(np.random.choice(len(weights), size=k, replace=True))
    
    return np.array(indices[:N])


def compute_ess(weights):
    """
    Compute Effective Sample Size.
    ESS = 1 / sum(w^2), normalized weights assumed.
    """
    return 1.0 / np.sum(weights ** 2)


def cov_ellipse(cov, q=0.95, npts=100):
    """
    Compute points for covariance ellipse at confidence level q.
    Returns array of (x, y) points centered at origin.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    
    angle = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
    scale = np.sqrt(stats.chi2.ppf(q, df=2))
    
    width = 2 * scale * np.sqrt(eigvals[0])
    height = 2 * scale * np.sqrt(eigvals[1])
    
    theta = np.linspace(0, 2 * np.pi, npts)
    ellipse = np.array([width/2 * np.cos(theta), height/2 * np.sin(theta)])
    
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    ellipse_rot = R @ ellipse
    
    return ellipse_rot.T


def circular_mean(angles, weights=None):
    """
    Compute circular mean of angles (in radians).
    Uses sin/cos averaging method.
    """
    if weights is None:
        weights = np.ones(len(angles)) / len(angles)
    
    sin_sum = np.sum(weights * np.sin(angles))
    cos_sum = np.sum(weights * np.cos(angles))
    
    return np.arctan2(sin_sum, cos_sum)


def weighted_covariance(particles, weights):
    """
    Compute weighted covariance matrix for x,y positions.
    particles: Nx3 array [x, y, theta]
    weights: N-array of normalized weights
    Returns: 2x2 covariance matrix
    """
    weights = weights / np.sum(weights)
    xy = particles[:, :2]
    mean = np.sum(weights[:, np.newaxis] * xy, axis=0)
    centered = xy - mean
    cov = np.zeros((2, 2))
    for i in range(len(weights)):
        cov += weights[i] * np.outer(centered[i], centered[i])
    return cov
