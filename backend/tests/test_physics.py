"""TEOS-10 physics unit tests (adapted to app.core.physics API)."""
import numpy as np

from app.core.physics import (
    compute_potential_density,
    compute_static_stability,
    compute_mld,
    validate_profile_physics,
)


def test_surface_potential_density_calculation():
    # Warm surface Arabian Sea water (28.5 C, 36.5 PSU)
    sigma = compute_potential_density(
        np.array([28.5]), np.array([36.5]), np.array([5.0])
    )
    assert 23.0 <= float(sigma[0]) <= 24.0, f"Surface density {sigma[0]} outside range"


def test_gravitational_stability_on_monotonic_profile():
    temps = np.array([28.5, 28.0, 26.0, 22.0, 18.0, 14.0, 11.0, 8.0])
    sals = np.array([36.5, 36.5, 36.2, 35.8, 35.5, 35.3, 35.2, 35.1])
    depths = np.array([5, 20, 50, 100, 150, 250, 500, 1000], dtype=float)

    sigma = compute_potential_density(temps, sals, depths)
    stability, is_stable, violations = compute_static_stability(sigma, depths)
    assert is_stable is True
    assert violations == 0
    assert float(np.min(stability)) >= 0.0


def test_density_inversion_detection():
    # Cold heavy water on top of warm light water -> unstable
    temps = np.array([20.0, 28.5])
    sals = np.array([36.5, 35.0])
    depths = np.array([5.0, 50.0])

    sigma = compute_potential_density(temps, sals, depths)
    stability, is_stable, violations = compute_static_stability(sigma, depths)
    assert is_stable is False
    assert violations >= 1
    assert float(np.min(stability)) < 0.0


def test_mld_detection():
    # Isothermal 28 C down to 30m, sharp drop at 50m
    sigma = np.array([23.4, 23.4, 23.4, 23.41, 23.6, 24.0])
    depths = np.array([5, 10, 20, 30, 50, 75], dtype=float)

    mld = compute_mld(sigma, depths)
    assert 30 <= mld <= 50, f"Computed MLD {mld} not matching thermocline shoaling"


def test_acceptance_profile_is_stable():
    # Reference profile that must pass the stability check
    result = validate_profile_physics(
        np.linspace(28, 7, 16), np.linspace(36.5, 35.2, 16)
    )
    assert result["is_gravitationally_stable"] is True
    assert result["stability_violations"] == 0
