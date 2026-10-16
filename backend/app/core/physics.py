"""
TEOS-10 Thermodynamic Calculations with Fallback

Implements potential density, static stability, Brunt-Väisälä frequency,
and Mixed Layer Depth using gsw (TEOS-10) with NumPy polynomial fallback.
"""
import numpy as np
from typing import Tuple, Dict, Optional, List
import warnings

# Standard 16 depth levels (dbar)
STANDARD_DEPTHS = np.array([5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)

# Fallback coefficients for potential density approximation
# σ_θ ≈ 28.14 - 0.0735*T - 0.00469*T^2 + 0.802*(S - 35.0)
# From UNESCO/TEOS-10 polynomial approximation
FALLBACK_COEFFS = {
    'const': 28.14,
    'T_linear': -0.0735,
    'T_quadratic': -0.00469,
    'S_linear': 0.802,
    'S_offset': 35.0
}

# Try to import gsw (TEOS-10 Gibbs SeaWater toolbox)
try:
    import gsw
    GSW_AVAILABLE = True
except (ImportError, AttributeError, OSError):
    GSW_AVAILABLE = False
    warnings.warn("gsw not available or C-extensions failed; using NumPy fallback for TEOS-10 calculations")


def compute_potential_density(temp: np.ndarray, sal: np.ndarray, 
                               pres: Optional[np.ndarray] = None,
                               lat: float = 20.0, lon: float = 65.0) -> np.ndarray:
    """
    Compute potential density anomaly σ_θ (kg/m³) from temperature and salinity.
    
    Uses gsw.sigma0 (TEOS-10) with NumPy polynomial fallback.
    
    Args:
        temp: Temperature array (°C) at standard depths
        sal: Salinity array (PSU) at standard depths
        pres: Pressure array (dbar) - defaults to STANDARD_DEPTHS
        lat: Latitude for SA/CT conversion (approximate)
        lon: Longitude for SA/CT conversion (approximate)
    
    Returns:
        Potential density anomaly σ_θ (kg/m³) relative to 1000 kg/m³
    """
    if pres is None:
        pres = STANDARD_DEPTHS
    
    temp = np.asarray(temp, dtype=float)
    sal = np.asarray(sal, dtype=float)
    pres = np.asarray(pres, dtype=float)
    
    if GSW_AVAILABLE:
        try:
            # Convert practical salinity to Absolute Salinity
            sa = gsw.SA_from_SP(sal, pres, lon, lat)
            # Convert in-situ temperature to Conservative Temperature
            ct = gsw.CT_from_t(sa, temp, pres)
            # Potential density anomaly referenced to 0 dbar
            sigma0 = gsw.sigma0(sa, ct)
            return sigma0
        except Exception:
            pass  # Fall through to fallback
    
    # NumPy polynomial fallback
    # σ_θ ≈ 28.14 - 0.0735*T - 0.00469*T^2 + 0.802*(S - 35.0)
    sigma = (FALLBACK_COEFFS['const'] + 
             FALLBACK_COEFFS['T_linear'] * temp + 
             FALLBACK_COEFFS['T_quadratic'] * temp**2 + 
             FALLBACK_COEFFS['S_linear'] * (sal - FALLBACK_COEFFS['S_offset']))
    return sigma


def compute_static_stability(sigma: np.ndarray, 
                              depths: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, int]:
    """
    Compute static gravitational stability ∂σ_θ/∂z.
    
    Physical stability requires ∂σ_θ/∂z ≥ 0 (density increases with depth).
    Uses forward differences (σ[i+1] - σ[i]) / (z[i+1] - z[i]) to match
    the physics loss function in training.
    
    Args:
        sigma: Potential density anomaly array (kg/m³)
        depths: Depth array (dbar) - defaults to STANDARD_DEPTHS
    
    Returns:
        (stability_array, is_stable, violation_count)
        - stability_array: ∂σ/∂z at each level (kg/m⁴) - length N-1 for N levels
        - is_stable: True if no violations
        - violation_count: Number of levels with ∂σ/∂z < -1e-6
    """
    if depths is None:
        depths = STANDARD_DEPTHS
    
    sigma = np.asarray(sigma, dtype=float)
    depths = np.asarray(depths, dtype=float)
    
    # Forward differences: (σ[i+1] - σ[i]) / (z[i+1] - z[i])
    # This matches the physics loss computation exactly
    d_sigma = sigma[1:] - sigma[:-1]  # (N-1,)
    d_depth = depths[1:] - depths[:-1]  # (N-1,)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        stability = np.where(d_depth != 0, d_sigma / d_depth, np.inf)
    
    # Violations: stability < -1e-6 (allow tiny numerical noise)
    violations = stability < -1e-6
    violation_count = int(np.sum(violations))
    is_stable = violation_count == 0
    
    return stability, is_stable, violation_count


def compute_brunt_vaisala(sigma: np.ndarray, 
                           depths: Optional[np.ndarray] = None,
                           g: float = 9.80665,
                           rho0: float = 1025.0) -> np.ndarray:
    """
    Compute Brunt-Väisälä buoyancy frequency squared N².
    
    N² = (g / ρ₀) * ∂σ_θ/∂z  [s⁻²]
    
    Args:
        sigma: Potential density anomaly (kg/m³)
        depths: Depth array (dbar)
        g: Gravitational acceleration (m/s²)
        rho0: Reference density (kg/m³)
    
    Returns:
        N² array (s⁻²), with negative values clipped to 0
        Length N-1 for N input levels (mid-point depths)
    """
    stability, _, _ = compute_static_stability(sigma, depths)
    # stability is d(sigma)/dz in kg m-3 per dbar (~ per metre), so N2 = g / rho0 * d(sigma)/dz
    N2 = (g / rho0) * stability
    return np.maximum(N2, 0.0)  # N² cannot be negative


def compute_mld(sigma: np.ndarray, 
                depths: Optional[np.ndarray] = None,
                threshold: float = 0.03) -> float:
    """
    Compute Mixed Layer Depth (MLD) using density threshold.
    
    MLD = min depth where σ_θ(z) - σ_θ(surface) ≥ threshold
    
    Args:
        sigma: Potential density anomaly (kg/m³)
        depths: Depth array (dbar)
        threshold: Density difference threshold (kg/m³)
    
    Returns:
        MLD in dbar, or NaN if not found
    """
    if depths is None:
        depths = STANDARD_DEPTHS
    
    sigma = np.asarray(sigma, dtype=float)
    
    if len(sigma) == 0 or np.isnan(sigma[0]):
        return np.nan
    
    surface_sigma = sigma[0]
    sigma_diff = sigma - surface_sigma
    
    # Find first depth where difference exceeds threshold
    mld_indices = np.where(sigma_diff >= threshold)[0]
    
    if len(mld_indices) > 0:
        return float(depths[mld_indices[0]])
    return np.nan


def compute_thermocline_gradient(temp: np.ndarray,
                                  depths: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """
    Compute maximum thermocline temperature gradient.
    
    Returns max |ΔT/Δz| and its depth index.
    """
    if depths is None:
        depths = STANDARD_DEPTHS
    
    temp = np.asarray(temp, dtype=float)
    depths = np.asarray(depths, dtype=float)
    
    dT = np.gradient(temp)
    dZ = np.gradient(depths)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        gradient = np.where(dZ != 0, np.abs(dT / dZ), 0.0)
    
    max_idx = int(np.nanargmax(gradient)) if not np.all(np.isnan(gradient)) else 0
    max_val = float(gradient[max_idx])
    
    return max_val, max_idx


def validate_profile_physics(temp: np.ndarray, sal: np.ndarray,
                              depths: Optional[np.ndarray] = None,
                              lat: float = 20.0, lon: float = 65.0) -> Dict:
    """
    Full physics validation for a temperature/salinity profile.
    
    Returns dict with all derived quantities and stability flags.
    """
    if depths is None:
        depths = STANDARD_DEPTHS
    
    # Potential density
    sigma = compute_potential_density(temp, sal, depths, lat, lon)
    
    # Static stability
    stability, is_stable, violation_count = compute_static_stability(sigma, depths)
    
    # Brunt-Väisälä
    N2 = compute_brunt_vaisala(sigma, depths)
    
    # MLD
    mld = compute_mld(sigma, depths)
    
    # Thermocline gradient
    max_grad, grad_idx = compute_thermocline_gradient(temp, depths)
    
    return {
        'potential_density': sigma,
        'stability': stability,
        'is_gravitationally_stable': is_stable,
        'stability_violations': violation_count,
        'brunt_vaisala_N2': N2,
        'mld_dbar': mld,
        'max_thermocline_gradient': max_grad,
        'thermocline_depth_idx': grad_idx,
        'engine': 'gsw' if GSW_AVAILABLE else 'fallback'
    }


# Test function for acceptance criteria
def test_acceptance_criteria() -> bool:
    """
    Test with the acceptance criteria profile:
    T = [28, 27, ..., 7] (16 values descending)
    S = [36.5, 36.4, ..., 35.2] (16 values descending)
    Should return is_gravitationally_stable = True, zero violations
    """
    T = np.linspace(28, 7, 16)
    S = np.linspace(36.5, 35.2, 16)
    
    result = validate_profile_physics(T, S)
    
    print(f"Engine: {result['engine']}")
    print(f"Is stable: {result['is_gravitationally_stable']}")
    print(f"Violations: {result['stability_violations']}")
    print(f"MLD: {result['mld_dbar']} dbar")
    print(f"Max N^2: {np.nanmax(result['brunt_vaisala_N2']):.2e} s^-2")
    print(f"Max thermocline grad: {result['max_thermocline_gradient']:.4f} C/dbar")
    
    assert result['is_gravitationally_stable'] == True, "Profile should be stable"
    assert result['stability_violations'] == 0, "Should have zero violations"
    
    print("OK Acceptance criteria PASSED")
    return True


if __name__ == "__main__":
    test_acceptance_criteria()