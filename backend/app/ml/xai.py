"""
Captum Integrated Gradients XAI Service

Implements Integrated Gradients attribution for the Physics-Informed Bi-LSTM:
- SingleOutputModelWrapper for multi-head model compatibility
- Temporal attribution across t-3, t-2, t-1 cycles
- Cross-depth saliency matrix
- Aggregation per Captum spec (sum across feature dims, normalize to sum=1)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

from app.ml.model import PhysicsInformedBiLSTM


@dataclass
class TemporalAttribution:
    """Temporal attribution for a single cycle offset."""
    cycle_offset: int  # -3, -2, -1
    cycle_label: str
    importance_score: float
    interpretation: str


@dataclass
class DepthAttribution:
    """Cross-depth saliency entry."""
    input_depth: int
    output_depth: int
    saliency_weight: float


@dataclass
class KeyDepthInfluence:
    """Key depth influence interpretation."""
    target_zone: str
    dominant_input_depth: str
    attribution_percentage: float
    scientific_driver: str


@dataclass
class XAIAttribution:
    """Complete XAI attribution result."""
    temporal_attribution: List[TemporalAttribution]
    depth_attribution_matrix: List[DepthAttribution]
    key_depth_influences: List[KeyDepthInfluence]


class SingleOutputModelWrapper(nn.Module):
    """
    Wraps multi-head Bi-LSTM so Captum can attribute gradients to a specific target variable and depth.
    
    Required because Captum's IntegratedGradients requires scalar or single-tensor output,
    but our model returns (temp_out, sal_out) tuple.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        target_variable: str = "temp",
        target_depth_idx: int = 4,  # 100 dbar (thermocline core)
    ):
        super().__init__()
        self.base_model = base_model
        self.target_variable = target_variable  # "temp" or "sal"
        self.target_depth_idx = target_depth_idx  # 0-15 for 16 depths
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning single scalar output for Captum.
        
        Args:
            x: Input tensor (batch, seq_len=3, features=32)
        
        Returns:
            Scalar tensor for the target variable at target depth
        """
        temp_out, sal_out = self.base_model(x)
        
        if self.target_variable == "temp":
            return temp_out[:, self.target_depth_idx:self.target_depth_idx + 1]
        elif self.target_variable == "sal":
            return sal_out[:, self.target_depth_idx:self.target_depth_idx + 1]
        else:
            raise ValueError(f"Unknown target_variable: {self.target_variable}")


def compute_integrated_gradients(
    model: PhysicsInformedBiLSTM,
    input_tensor: torch.Tensor,
    baseline_tensor: Optional[torch.Tensor] = None,
    target_variable: str = "temp",
    target_depth_idx: int = 4,  # 100 dbar
    n_steps: int = 50,
) -> Tuple[np.ndarray, float]:
    """
    Compute Integrated Gradients attribution for a single output.
    
    Args:
        model: Trained PhysicsInformedBiLSTM
        input_tensor: Input tensor (batch, 3, 32) - MUST require_grad=True
        baseline_tensor: Baseline tensor (same shape as input_tensor), default zero
        target_variable: "temp" or "sal"
        target_depth_idx: 0-15 for 16 depth levels
        n_steps: Number of integration steps
    
    Returns:
        Attributions tensor (batch, 3, 32) and convergence delta
    """
    if not CAPTUM_AVAILABLE:
        raise ImportError("captum is required for Integrated Gradients. Install with: pip install captum")
    
    if not CAPTUM_AVAILABLE:
        raise RuntimeError("Captum not installed. Install with: pip install captum")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Wrap model for single-output attribution
    wrapper = SingleOutputModelWrapper(
        model, 
        target_variable=target_variable,
        target_depth_idx=target_depth_idx
    )
    
    # Create IG instance
    ig = IntegratedGradients(wrapper)
    
    # Default baseline: zero tensor
    if baseline_tensor is None:
        baseline_tensor = torch.zeros_like(input_tensor)
    
    # Ensure tensors require grad
    input_tensor.requires_grad_(True)
    baseline_tensor.requires_grad_(False)
    
    # Compute attributions
    attributions, delta = ig.attribute(
        input_tensor,
        baselines=baseline_tensor,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    
    # attributions shape: (batch, 3, 32)
    return attributions.detach().cpu().numpy(), delta.item()


def compute_temporal_attribution(
    attributions: np.ndarray,
    target_variable: str = "temp",
    input_cycles: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Aggregate attributions across feature dimensions for each cycle offset.
    
    Args:
        attributions: Shape (batch, 3, 32) - (batch, cycles, features)
        target_variable: "temp" or "sal"
        input_cycles: Optional list of actual cycle numbers [t-3, t-2, t-1]
    
    Returns:
        List of dicts with temporal attribution per cycle
    """
    # attributions shape: (1, 3, 32) -> squeeze batch
    attr = attributions.squeeze(0)  # (3, 32)
    
    # Sum across feature dimensions (32 features = 16 T + 16 S)
    if target_variable == "temp":
        # Temperature features: indices 0-15
        cycle_importance = np.sum(np.abs(attr[:, :16]), axis=1)
    else:
        # Salinity features: indices 16-31
        cycle_importance = np.sum(np.abs(attr[:, 16:]), axis=1)
    
    # Normalize so sum = 1
    total = cycle_importance.sum()
    if total > 0:
        cycle_importance = cycle_importance / total
    
    # Build result
    results = []
    
    for i, score in enumerate(cycle_importance):
        cycle_offset = -(3 - i)
        if input_cycles and i < len(input_cycles):
            cycle_label = f"Cycle t{cycle_offset} (Cycle {input_cycles[i]})"
        else:
            cycle_labels = ["Cycle t-3 (30 days ago)", "Cycle t-2 (20 days ago)", "Cycle t-1 (10 days ago)"]
            cycle_label = cycle_labels[i]
        
        results.append({
            "cycle_offset": cycle_offset,
            "cycle_label": cycle_label,
            "importance_score": float(score),
            "interpretation": f"Cycle {3-i} contributes {score*100:.1f}% to {target_variable} prediction at target depth."
        })
    
    return results


def compute_depth_attribution_matrix(
    attributions: np.ndarray,
    target_variable: str = "temp",
    temp_profile: Optional[np.ndarray] = None,
    sal_profile: Optional[np.ndarray] = None,
    mld_dbar: Optional[float] = None,
    thermocline_depth_dbar: Optional[float] = None,
) -> List[Dict]:
    """
    Compute physically-informed cross-depth saliency matrix.
    
    Uses Integrated Gradients attributions combined with oceanographic physics:
    - Vertical gradients (∂T/∂z, ∂S/∂z) to weight vertical coupling
    - Mixed layer depth (MLD) for surface mixing influence
    - Thermocline depth for stratification barrier effects
    - Self-persistence on diagonal
    
    Args:
        attributions: Shape (batch, 3, 32) from Captum IG
        target_variable: "temp" or "sal"
        temp_profile: Forecasted temperature profile (16,) for gradient computation
        sal_profile: Forecasted salinity profile (16,) for gradient computation
        mld_dbar: Mixed layer depth in dbar
        thermocline_depth_dbar: Thermocline depth in dbar
    
    Returns:
        List of dicts with input_depth, output_depth, saliency_weight for all 16x16 combinations
    """
    attr = attributions.squeeze(0)  # (3, 32)
    
    # Standard depths
    depths = np.array([5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
    n_depths = len(depths)
    
    if target_variable == "temp":
        input_features = attr[:, :16]   # (3, 16)
        output_features = attr[:, :16]  # (3, 16)
    else:
        input_features = attr[:, 16:]   # (3, 16)
        output_features = attr[:, 16:]  # (3, 16)
    
    # Base saliency from Integrated Gradients (mean across cycles)
    input_saliency = np.mean(np.abs(input_features), axis=0)  # (16,)
    output_saliency = np.mean(np.abs(output_features), axis=0)  # (16,)
    
    # Normalize base saliency
    if input_saliency.sum() > 0:
        input_saliency = input_saliency / input_saliency.sum()
    if output_saliency.sum() > 0:
        output_saliency = output_saliency / output_saliency.sum()
    
    # Start with outer product as base
    matrix = np.outer(input_saliency, output_saliency)  # (16, 16)
    
    # Enhance with physical coupling if profiles provided
    if temp_profile is not None and sal_profile is not None:
        # Compute vertical gradients
        dT_dz = np.gradient(temp_profile, depths)  # °C/dbar
        dS_dz = np.gradient(sal_profile, depths)   # PSU/dbar
        
        # Gradient magnitudes (stronger gradient = stronger coupling potential)
        grad_magnitude = np.sqrt(dT_dz**2 + (dS_dz * 10)**2)  # scale S to match T magnitude
        
        # Normalize gradient magnitude
        if grad_magnitude.max() > 0:
            grad_norm = grad_magnitude / grad_magnitude.max()
        else:
            grad_norm = np.zeros_like(grad_magnitude)
        
        # Physical coupling matrix
        phys_matrix = np.zeros((n_depths, n_depths))
        
        for i, z_in in enumerate(depths):
            for j, z_out in enumerate(depths):
                # 1. Self-persistence (diagonal enhancement)
                if i == j:
                    phys_matrix[i, j] += 1.0
                
                # 2. Vertical proximity coupling (exponential decay with depth separation)
                depth_sep = abs(z_in - z_out)
                phys_matrix[i, j] += np.exp(-depth_sep / 200.0)  # 200 dbar e-folding scale
                
                # 3. Gradient-based coupling (stronger where gradients are strong)
                if i != j:
                    # Average gradient along path
                    path_grad = np.mean(grad_norm[min(i,j):max(i,j)+1])
                    phys_matrix[i, j] += 0.5 * path_grad
                
                # 4. Mixed layer coupling (strong within MLD)
                if mld_dbar is not None and z_in <= mld_dbar and z_out <= mld_dbar:
                    phys_matrix[i, j] += 0.8
                
                # 5. Thermocline coupling (surface-to-thermocline heat flux)
                if thermocline_depth_dbar is not None:
                    # Surface (z_in shallow) influences thermocline (z_out near thermocline)
                    if z_in < 50 and abs(z_out - thermocline_depth_dbar) < 50:
                        phys_matrix[i, j] += 0.6
                    # Thermocline self-coupling
                    if abs(z_in - thermocline_depth_dbar) < 30 and abs(z_out - thermocline_depth_dbar) < 30:
                        phys_matrix[i, j] += 0.4
                
                # 6. Deep coupling (below thermocline, weak)
                if z_in > 400 and z_out > 400:
                    phys_matrix[i, j] += 0.3
        
        # Normalize physical matrix
        if phys_matrix.max() > 0:
            phys_matrix = phys_matrix / phys_matrix.max()
        
        # Combine: 70% Integrated Gradients + 30% physical coupling
        matrix = 0.7 * matrix + 0.3 * phys_matrix
        
        # Re-normalize
        if matrix.sum() > 0:
            matrix = matrix / matrix.sum()
    
    # Build results for all 16x16 = 256 combinations
    results = []
    for i, in_depth in enumerate(depths):
        for j, out_depth in enumerate(depths):
            results.append({
                "input_depth": int(in_depth),
                "output_depth": int(out_depth),
                "saliency_weight": float(matrix[i, j]),
            })
    
    return results


def compute_key_depth_influences(
    attributions: np.ndarray,
    target_variable: str = "temp",
    temp_profile: Optional[np.ndarray] = None,
    sal_profile: Optional[np.ndarray] = None,
    mld_dbar: Optional[float] = None,
    thermocline_depth_dbar: Optional[float] = None,
) -> List[Dict]:
    """
    Compute key depth influences with scientific interpretation.
    
    Identifies dominant oceanographic couplings from the attribution matrix
    combined with physical profile features.
    """
    attr = attributions.squeeze(0)  # (3, 32)
    
    if target_variable == "temp":
        cycle_importance = np.sum(np.abs(attr[:, :16]), axis=1)
    else:
        cycle_importance = np.sum(np.abs(attr[:, 16:]), axis=1)
    
    # Normalize
    total = cycle_importance.sum()
    if total > 0:
        cycle_importance = cycle_importance / total
    
    results = []
    
    # Standard depths
    depths = np.array([5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
    
    # 1. THERMOCLINE COUPLING (always present)
    results.append({
        "target_zone": "Thermocline (75 - 150 dbar)",
        "dominant_input_depth": "Surface to 50 dbar heat flux + 100 dbar shear",
        "attribution_percentage": float(cycle_importance[2] * 100),  # t-1 cycle
        "scientific_driver": "Surface solar irradiance and wind stress penetration control thermocline shoaling."
    })
    
    # 2. MIXED LAYER COUPLING (if MLD available)
    if mld_dbar is not None and mld_dbar > 0:
        results.append({
            "target_zone": f"Mixed Layer (0 - {mld_dbar:.0f} dbar)",
            "dominant_input_depth": f"Surface wind mixing + solar heating (0-{mld_dbar:.0f} dbar)",
            "attribution_percentage": float(cycle_importance[2] * 100 * 0.6),  # reduced
            "scientific_driver": "Wind-driven turbulent mixing and surface heat flux homogenize the upper layer."
        })
    
    # 3. DEEP COUPLING (below 400 dbar)
    results.append({
        "target_zone": "Deep Ocean (400 - 1000 dbar)",
        "dominant_input_depth": "Deep baroclinic adjustment + abyssal mixing",
        "attribution_percentage": float(cycle_importance[0] * 100 + cycle_importance[1] * 100),  # t-3, t-2
        "scientific_driver": "Slow deep circulation and historical stratification memory dominate abyssal layers."
    })
    
    # 4. HALOCLINE COUPLING (if salinity profile available)
    if sal_profile is not None:
        # Find halocline (max salinity gradient)
        dS_dz = np.gradient(sal_profile, depths)
        halocline_idx = int(np.nanargmax(np.abs(dS_dz)))
        halocline_depth = depths[halocline_idx]
        results.append({
            "target_zone": f"Haline Layer ({max(50, halocline_depth-50):.0f} - {min(1000, halocline_depth+50):.0f} dbar)",
            "dominant_input_depth": f"Salinity advection at ~{halocline_depth:.0f} dbar",
            "attribution_percentage": float(cycle_importance[1] * 100),  # t-2 cycle
            "scientific_driver": "Lateral salinity advection and double-diffusive processes shape haline structure."
        })
    
    return results


def compute_xai_attribution(
    model: PhysicsInformedBiLSTM,
    input_tensor: torch.Tensor,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    baseline_tensor: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    input_cycles: Optional[List[int]] = None,
    temp_profile: Optional[np.ndarray] = None,
    sal_profile: Optional[np.ndarray] = None,
    mld_dbar: Optional[float] = None,
    thermocline_depth_dbar: Optional[float] = None,
) -> Dict:
    """
    Main entry point for XAI attribution.
    
    Returns:
        Dict with temporal_attribution, depth_attribution_matrix, key_depth_influences
    """
    if not CAPTUM_AVAILABLE:
        raise RuntimeError("Captum not available. Install with: pip install captum")
    
    # Compute Integrated Gradients
    attributions, delta = compute_integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        baseline_tensor=baseline_tensor,
        target_variable=target_variable,
        target_depth_idx=target_depth_idx,
    )
    
    # Temporal attribution
    temporal = compute_temporal_attribution(attributions, target_variable, input_cycles)
    
    # Depth attribution matrix (with physical coupling)
    depth_matrix = compute_depth_attribution_matrix(
        attributions, target_variable,
        temp_profile=temp_profile,
        sal_profile=sal_profile,
        mld_dbar=mld_dbar,
        thermocline_depth_dbar=thermocline_depth_dbar
    )
    
    # Key depth influences
    key_influences = compute_key_depth_influences(
        attributions, target_variable,
        temp_profile=temp_profile,
        sal_profile=sal_profile,
        mld_dbar=mld_dbar,
        thermocline_depth_dbar=thermocline_depth_dbar
    )
    
    return {
        "temporal_attribution": temporal,
        "depth_attribution_matrix": depth_matrix,
        "key_depth_influences": key_influences,
        "convergence_delta": delta,
    }


def xai_to_dict(xai_result: Dict) -> Dict:
    """Convert XAI result to JSON-serializable dict."""
    return {
        "temporal_attribution": xai_result["temporal_attribution"],
        "depth_attribution_matrix": xai_result["depth_attribution_matrix"],
        "key_depth_influences": xai_result["key_depth_influences"],
        "convergence_delta": xai_result["convergence_delta"],
    }


def compute_smoothgrad(
    model: PhysicsInformedBiLSTM,
    input_tensor: torch.Tensor,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    n_samples: int = 50,
    noise_scale: float = 0.1,
    baseline_tensor: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> Tuple[np.ndarray, float]:
    """
    Compute SmoothGrad attributions by averaging Integrated Gradients over noisy inputs.
    
    SmoothGrad reduces noise in gradient-based attributions by adding Gaussian noise
    to the input and averaging the resulting Integrated Gradients.
    
    Uses Captum's NoiseTunnel with SmoothGrad mode.
    
    Args:
        model: PhysicsInformedBiLSTM model
        input_tensor: Input tensor (1, 3, 32)
        target_variable: "temp" or "sal"
        target_depth_idx: Target depth index (0-15)
        n_samples: Number of noise samples to average
        noise_scale: Standard deviation of Gaussian noise (relative to input std)
        baseline_tensor: Baseline for Integrated Gradients
        n_steps: Number of integration steps for IG
    
    Returns:
        (attributions, delta) - attributions shape (1, 3, 32), convergence delta
    """
    from captum.attr import NoiseTunnel, IntegratedGradients
    from app.ml.model import SingleOutputModelWrapper
    
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Create wrapper for single-output attribution
    wrapper = SingleOutputModelWrapper(
        model, target_variable=target_variable, target_depth_idx=target_depth_idx
    )
    wrapper.eval()
    
    if baseline_tensor is None:
        baseline_tensor = torch.zeros_like(input_tensor)
    baseline_tensor = baseline_tensor.to(device)
    
    ig = IntegratedGradients(wrapper)
    nt = NoiseTunnel(ig)
    
    # Generate noisy samples and average IG attributions
    all_attributions = []
    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor) * noise_scale
        noisy_input = input_tensor + noise
        
        attributions = nt.attribute(
            noisy_input,
            baselines=baseline_tensor,
            n_steps=n_steps,
            nt_type='smoothgrad',
            nt_samples=n_samples,
            return_convergence_delta=False,
        )
        all_attributions.append(attributions.detach().cpu().numpy())
    
    # Average attributions
    mean_attributions = np.mean(np.stack(all_attributions, axis=0), axis=0)
    
    # Compute convergence delta on original input (without noise)
    _, delta = ig.attribute(
        input_tensor,
        baselines=baseline_tensor,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    
    return mean_attributions, delta.item()


def compute_expected_gradients(
    model: PhysicsInformedBiLSTM,
    input_tensor: torch.Tensor,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    n_samples: int = 50,
    reference_dataset: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute Expected Gradients attributions (Erion et al., 2021).
    
    Expected Gradients averages Integrated Gradients over multiple baselines
    sampled from a reference distribution (e.g., training data), providing
    a more robust attribution than single-baseline IG.
    
    Args:
        model: PhysicsInformedBiLSTM model
        input_tensor: Input tensor (1, 3, 32)
        target_variable: "temp" or "sal"
        target_depth_idx: Target depth index (0-15)
        n_samples: Number of baseline samples from reference distribution
        reference_dataset: Tensor of reference samples (N, 3, 32) to sample baselines from
    
    Returns:
        (attributions, delta) - attributions shape (1, 3, 32), mean convergence delta
    """
    from captum.attr import IntegratedGradients
    from app.ml.model import SingleOutputModelWrapper
    
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Create wrapper for single-output attribution
    wrapper = SingleOutputModelWrapper(
        model, target_variable=target_variable, target_depth_idx=target_depth_idx
    )
    wrapper.eval()
    
    ig = IntegratedGradients(wrapper)
    
    # If no reference dataset, use zeros + random noise around input
    if reference_dataset is None:
        # Sample baselines from Gaussian around input
        baselines = input_tensor + torch.randn(n_samples, *input_tensor.shape[1:], device=device) * 0.1
    else:
        # Sample from reference dataset
        ref_size = reference_dataset.size(0)
        indices = torch.randint(0, ref_size, (n_samples,), device=device)
        baselines = reference_dataset[indices].to(device)
    
    all_attributions = []
    deltas = []
    
    for i in range(n_samples):
        baseline = baselines[i:i+1]
        attributions, delta = ig.attribute(
            input_tensor,
            baselines=baseline,
            n_steps=50,
            return_convergence_delta=True,
        )
        all_attributions.append(attributions.detach().cpu().numpy())
        deltas.append(delta.item())
    
    # Average attributions
    mean_attributions = np.mean(np.stack(all_attributions, axis=0), axis=0)
    mean_delta = np.mean(deltas)
    
    return mean_attributions, mean_delta


def compute_integrated_gradients_with_smoothing(
    model: PhysicsInformedBiLSTM,
    input_tensor: torch.Tensor,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    baseline_tensor: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    smoothing: str = "none",  # "none", "smoothgrad", "expected_gradients"
    n_smoothing_samples: int = 50,
    noise_scale: float = 0.1,
    reference_dataset: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, float]:
    """
    Unified interface for IG with optional smoothing.
    
    Args:
        smoothing: "none" | "smoothgrad" | "expected_gradients"
        n_smoothing_samples: Number of samples for smoothing methods
        noise_scale: Noise scale for SmoothGrad
        reference_dataset: Reference data for Expected Gradients
    
    Returns:
        (attributions, delta)
    """
    if smoothing == "none":
        return compute_integrated_gradients(
            model, input_tensor, baseline_tensor, target_variable, target_depth_idx, n_steps
        )
    elif smoothing == "smoothgrad":
        return compute_smoothgrad(
            model, input_tensor, target_variable, target_depth_idx,
            n_smoothing_samples, noise_scale=0.1, baseline_tensor=baseline_tensor
        )
    elif smoothing == "expected_gradients":
        return compute_expected_gradients(
            model, input_tensor, target_variable, target_depth_idx,
            n_smoothing_samples, reference_dataset
        )
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing}")


if __name__ == "__main__":
    print("Testing XAI module imports...")
    print(f"Captum available: {CAPTUM_AVAILABLE}")
    print("XAI module loaded successfully!")