"""
Particle Filter Project
A complete implementation of particle filter for robot localization.
"""

__version__ = "1.0.0"
__author__ = "Particle Filter Team"

from .particle_filter import ParticleFilter, RobotSim
from .utils import (
    wrap_angle,
    gaussian_pdf,
    systematic_resample,
    residual_resample,
    compute_ess,
    circular_mean,
    weighted_covariance,
    cov_ellipse
)

__all__ = [
    'ParticleFilter',
    'RobotSim',
    'wrap_angle',
    'gaussian_pdf',
    'systematic_resample',
    'residual_resample',
    'compute_ess',
    'circular_mean',
    'weighted_covariance',
    'cov_ellipse'
]
