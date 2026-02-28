"""
Unit tests for particle filter components.
Tests core functions: wrap_angle, gaussian_pdf, resampling, weight normalization.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (wrap_angle, gaussian_pdf, systematic_resample, residual_resample,
                   compute_ess, circular_mean, weighted_covariance)


def test_wrap_angle():
    """Test angle normalization to [-pi, pi]."""
    print("Testing wrap_angle...")
    
    # Test basic cases
    assert np.isclose(wrap_angle(0), 0), "wrap_angle(0) should be 0"
    assert np.isclose(abs(wrap_angle(np.pi)), np.pi), "wrap_angle(pi) should be ±pi"
    assert np.isclose(abs(wrap_angle(-np.pi)), np.pi), "wrap_angle(-pi) should be ±pi"
    
    # Test wrapping
    assert np.isclose(wrap_angle(2 * np.pi), 0), "wrap_angle(2*pi) should wrap to 0"
    assert np.isclose(abs(wrap_angle(3 * np.pi)), np.pi), "wrap_angle(3*pi) should wrap to ±pi"
    assert np.isclose(abs(wrap_angle(-3 * np.pi)), np.pi), "wrap_angle(-3*pi) should wrap to ±pi"
    
    # Test array input
    angles = np.array([0, np.pi, 2*np.pi, -np.pi, 3*np.pi])
    wrapped = wrap_angle(angles)
    assert len(wrapped) == len(angles), "Should handle arrays"
    assert all(-np.pi <= a <= np.pi for a in wrapped), "All angles should be in [-pi, pi]"
    
    print("  ✓ All wrap_angle tests passed")


def test_gaussian_pdf():
    """Test Gaussian PDF computation."""
    print("Testing gaussian_pdf...")
    
    # PDF at mean should be maximum
    pdf_at_mean = gaussian_pdf(0, 0, 1)
    pdf_away = gaussian_pdf(1, 0, 1)
    assert pdf_at_mean > pdf_away, "PDF should be maximum at mean"
    
    # Check specific value
    expected = 1 / np.sqrt(2 * np.pi)
    actual = gaussian_pdf(0, 0, 1)
    assert np.isclose(actual, expected, rtol=1e-5), f"PDF(0, 0, 1) should be {expected}"
    
    # Test symmetry
    pdf_plus = gaussian_pdf(1, 0, 1)
    pdf_minus = gaussian_pdf(-1, 0, 1)
    assert np.isclose(pdf_plus, pdf_minus), "Gaussian should be symmetric"
    
    print("  ✓ All gaussian_pdf tests passed")


def test_systematic_resample():
    """Test systematic resampling."""
    print("Testing systematic_resample...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Test with uniform weights
    N = 100
    weights = np.ones(N) / N
    indices = systematic_resample(weights, N)
    
    assert len(indices) == N, f"Should return N={N} indices"
    assert all(0 <= i < N for i in indices), "All indices should be in valid range"
    
    # Test with skewed weights (one particle has high weight)
    weights = np.ones(N) / (N * 10)
    weights[0] = 0.9
    weights /= np.sum(weights)  # Normalize
    
    indices = systematic_resample(weights, N)
    assert len(indices) == N, "Should still return N indices"
    
    # High-weight particle should be sampled more
    count_high = np.sum(indices == 0)
    assert count_high > N / 10, "High-weight particle should be sampled frequently"
    
    print("  ✓ All systematic_resample tests passed")


def test_residual_resample():
    """Test residual resampling."""
    print("Testing residual_resample...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Test with uniform weights
    N = 100
    weights = np.ones(N) / N
    indices = residual_resample(weights, N)
    
    assert len(indices) == N, f"Should return N={N} indices"
    assert all(0 <= i < N for i in indices), "All indices should be in valid range"
    
    # Test with exact integer copies
    weights = np.array([0.5, 0.3, 0.2])
    N = 10
    indices = residual_resample(weights, N)
    
    assert len(indices) == N, "Should return exactly N indices"
    
    print("  ✓ All residual_resample tests passed")


def test_weight_normalization():
    """Test that weights sum to 1 after normalization."""
    print("Testing weight normalization...")
    
    # Create random weights
    np.random.seed(42)
    weights = np.random.rand(100)
    
    # Normalize
    weights /= np.sum(weights)
    
    assert np.isclose(np.sum(weights), 1.0), "Normalized weights should sum to 1"
    assert all(w >= 0 for w in weights), "All weights should be non-negative"
    
    print("  ✓ All weight_normalization tests passed")


def test_compute_ess():
    """Test Effective Sample Size computation."""
    print("Testing compute_ess...")
    
    # Uniform weights: ESS should equal N
    N = 100
    weights = np.ones(N) / N
    ess = compute_ess(weights)
    assert np.isclose(ess, N), f"ESS with uniform weights should be {N}"
    
    # Single particle with weight 1: ESS should be 1
    weights = np.zeros(N)
    weights[0] = 1.0
    ess = compute_ess(weights)
    assert np.isclose(ess, 1.0), "ESS with single particle should be 1"
    
    # Intermediate case
    weights = np.array([0.5, 0.5])
    ess = compute_ess(weights)
    assert 1 <= ess <= 2, "ESS should be between 1 and N"
    
    print("  ✓ All compute_ess tests passed")


def test_circular_mean():
    """Test circular mean computation."""
    print("Testing circular_mean...")
    
    # Mean of 0 and 0 should be 0
    angles = np.array([0, 0])
    mean = circular_mean(angles)
    assert np.isclose(mean, 0), "Mean of [0, 0] should be 0"
    
    # Mean of 0 and pi should be pi/2 (approximately)
    angles = np.array([0, np.pi/2])
    mean = circular_mean(angles)
    assert np.isclose(mean, np.pi/4), "Mean of [0, pi/2] should be pi/4"
    
    # Test with weights
    angles = np.array([0, np.pi])
    weights = np.array([0.9, 0.1])
    mean = circular_mean(angles, weights)
    # Should be closer to 0 than to pi
    assert abs(mean) < abs(mean - np.pi), "Weighted mean should favor higher-weight angle"
    
    print("  ✓ All circular_mean tests passed")


def test_weighted_covariance():
    """Test weighted covariance computation."""
    print("Testing weighted_covariance...")
    
    # Simple case: two particles at (0,0) and (1,1)
    particles = np.array([[0, 0, 0], [1, 1, 0]])
    weights = np.array([0.5, 0.5])
    
    cov = weighted_covariance(particles, weights)
    
    assert cov.shape == (2, 2), "Covariance should be 2x2"
    assert cov[0, 0] >= 0, "Variance should be non-negative"
    assert cov[1, 1] >= 0, "Variance should be non-negative"
    
    # Covariance should be symmetric
    assert np.isclose(cov[0, 1], cov[1, 0]), "Covariance matrix should be symmetric"
    
    print("  ✓ All weighted_covariance tests passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80 + "\n")
    
    try:
        test_wrap_angle()
        test_gaussian_pdf()
        test_systematic_resample()
        test_residual_resample()
        test_weight_normalization()
        test_compute_ess()
        test_circular_mean()
        test_weighted_covariance()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("="*80 + "\n")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("="*80 + "\n")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
