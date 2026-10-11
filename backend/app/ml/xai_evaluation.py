"""
XAI Faithfulness & Quality Evaluation Metrics for FloatChat

Implements standard explainability evaluation metrics:
1. Insertion/Deletion curves (Petsiuk et al.)
2. Faithfulness correlation (attribution vs. model sensitivity)
3. Stability metrics (noise robustness, local Lipschitz)
4. Domain-specific physical plausibility checks
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


@dataclass
class InsertionDeletionResult:
    """Result of insertion/deletion curve evaluation."""
    insertion_auc: float      # Area under insertion curve (higher = better)
    deletion_auc: float       # Area under deletion curve (lower = better)
    insertion_curve: np.ndarray  # Prediction vs. fraction of features inserted
    deletion_curve: np.ndarray   # Prediction vs. fraction of features deleted
    feature_order: np.ndarray    # Indices of features in attribution order


@dataclass
class FaithfulnessResult:
    """Result of faithfulness correlation evaluation."""
    pearson_r: float          # Pearson correlation
    spearman_r: float         # Spearman rank correlation
    p_value: float            # Significance
    n_samples: int


@dataclass
class StabilityResult:
    """Result of stability evaluation."""
    mean_cosine_similarity: float  # Mean cosine sim between noisy attributions
    std_cosine_similarity: float
    local_lipschitz_estimate: float


def compute_insertion_deletion_curve(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attributions: np.ndarray,
    target_variable: str = "temp",
    n_steps: int = 20,
    baseline: Optional[torch.Tensor] = None,
    climatology: Optional[torch.Tensor] = None,
    mode: str = "insertion",
) -> InsertionDeletionResult:
    """
    Compute insertion/deletion curve for feature attribution faithfulness.
    
    Args:
        model: Neural network model
        input_tensor: Input tensor (1, seq_len=3, features=32)
        attributions: Attribution scores (1, 3, 32) or flattened
        target_variable: "temp" or "sal" (which output head)
        n_steps: Number of steps for curve
        baseline: Baseline tensor (default: climatology or zeros)
        climatology: Climatological mean profile for realistic baseline
        mode: "insertion" or "deletion"
    
    Returns:
        InsertionDeletionResult with AUC and curves
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Use climatology as baseline if provided, else zeros
    if baseline is None:
        if climatology is not None:
            baseline = climatology.to(device)
        else:
            baseline = torch.zeros_like(input_tensor)
    
    input_tensor = input_tensor.to(device)
    baseline = baseline.to(device)
    
    # Flatten attributions to 1D for ordering (per-feature, not grouped)
    attr_flat = attributions.flatten()
    feature_order = np.argsort(-attr_flat)  # Descending (most important first)
    
    n_features = len(attr_flat)
    step_size = max(1, n_features // n_steps)
    
    curve_values = []
    
    with torch.no_grad():
        for i in range(0, n_features + 1, step_size):
            if mode == "insertion":
                # Start from baseline, insert top features
                mask = torch.zeros(n_features, device=device)
                mask[feature_order[:i]] = 1.0
                masked_input = baseline + mask.reshape(input_tensor.shape) * (input_tensor - baseline)
            else:  # deletion
                # Start from input, remove top features
                mask = torch.ones(n_features, device=device)
                mask[feature_order[:i]] = 0.0
                masked_input = mask.reshape(input_tensor.shape) * input_tensor + \
                               (1 - mask.reshape(input_tensor.shape)) * baseline
            
            output = model(masked_input)
            if isinstance(output, tuple):
                # target_variable is "temp" (index 0) or "sal" (index 1)
                output = output[0 if target_variable == "temp" else 1]
            pred = output[0, 0].item() if output.dim() > 1 else output.item()
            curve_values.append(pred)
    
    curve = np.array(curve_values)
    x_vals = np.linspace(0, 1, len(curve))
    auc = np.trapz(curve, x_vals)
    
    return InsertionDeletionResult(
        insertion_auc=auc if mode == "insertion" else 0.0,
        deletion_auc=auc if mode == "deletion" else 0.0,
        insertion_curve=curve if mode == "insertion" else np.array([]),
        deletion_curve=curve if mode == "deletion" else np.array([]),
        feature_order=feature_order,
    )


def compute_faithfulness_correlation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attributions: np.ndarray,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    n_perturbations: int = 100,
    noise_scale: float = 0.01,
) -> FaithfulnessResult:
    """
    Compute correlation between attributions and actual model sensitivity.
    
    Uses SingleOutputModelWrapper to match the attribution target (specific depth).
    Perturbs each feature and measures actual scalar output change vs. predicted (attribution).
    """
    from app.ml.model import SingleOutputModelWrapper
    
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Create wrapper matching the attribution target
    wrapper = SingleOutputModelWrapper(model, target_variable=target_variable, target_depth_idx=target_depth_idx)
    wrapper.eval()
    
    # Get baseline prediction (scalar)
    with torch.no_grad():
        baseline_val = wrapper(input_tensor).item()
    
    n_features = input_tensor.numel()
    attr_flat = attributions.flatten()
    
    actual_sensitivities = []
    pred_sensitivities = []
    
    # Sample random features for efficiency
    feature_indices = np.random.choice(n_features, min(n_perturbations, n_features), replace=False)
    
    # Input tensor is (1, 3, 32) -> [batch, cycle, feature]
    # Feature index mapping: idx -> (cycle_idx, feat_idx) where cycle_idx = idx // 32, feat_idx = idx % 32
    
    for idx in feature_indices:
        # Perturb feature using direct indexing (input tensor is non-contiguous)
        perturbed = input_tensor.clone()
        cycle_idx = idx // 32
        feat_idx = idx % 32
        original_val = perturbed[0, cycle_idx, feat_idx].item()
        perturbed[0, cycle_idx, feat_idx] = original_val + noise_scale * (abs(original_val) + 1e-6)
        
        with torch.no_grad():
            perturbed_val = wrapper(perturbed).item()
        
        actual_sens = perturbed_val - baseline_val
        pred_sens = attr_flat[idx]
        
        actual_sensitivities.append(actual_sens)
        pred_sensitivities.append(pred_sens)
    
    # Compute correlations
    pearson_r, p_pearson = stats.pearsonr(pred_sensitivities, actual_sensitivities)
    spearman_r, p_spearman = stats.spearmanr(pred_sensitivities, actual_sensitivities)
    
    return FaithfulnessResult(
        pearson_r=pearson_r,
        spearman_r=spearman_r,
        p_value=min(p_pearson, p_spearman),
        n_samples=len(feature_indices),
    )


def compute_stability_metrics(
    model: nn.Module,
    input_tensor: torch.Tensor,
    compute_attributions_fn: Callable,
    n_noise_trials: int = 20,
    noise_scale: float = 0.01,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    **attribution_kwargs,
) -> StabilityResult:
    """
    Compute stability of attributions under input noise.
    
    Measures how much attributions change when small noise is added to input.
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Get baseline attributions
    base_attr = compute_attributions_fn(model, input_tensor, target_variable=target_variable, target_depth_idx=target_depth_idx, **attribution_kwargs)
    if isinstance(base_attr, dict):
        base_attr = base_attr.get("attributions", base_attr)
    base_flat = base_attr.flatten()
    
    similarities = []
    lipschitz_estimates = []
    
    for _ in range(n_noise_trials):
        # Add noise
        noise = torch.randn_like(input_tensor) * noise_scale
        noisy_input = input_tensor + noise
        
        # Compute attributions for noisy input
        noisy_attr = compute_attributions_fn(model, noisy_input, target_variable=target_variable, target_depth_idx=target_depth_idx, **attribution_kwargs)
        if isinstance(noisy_attr, dict):
            noisy_attr = noisy_attr.get("attributions", noisy_attr)
        noisy_flat = noisy_attr.flatten()
        
        # Cosine similarity
        cos_sim = np.dot(base_flat, noisy_flat) / (np.linalg.norm(base_flat) * np.linalg.norm(noisy_flat) + 1e-8)
        similarities.append(cos_sim)
        
        # Local Lipschitz estimate
        attr_diff = np.linalg.norm(base_flat - noisy_flat)
        input_diff = torch.norm(noise).item()
        if input_diff > 0:
            lipschitz_estimates.append(attr_diff / input_diff)
    
    return StabilityResult(
        mean_cosine_similarity=float(np.mean(similarities)),
        std_cosine_similarity=float(np.std(similarities)),
        local_lipschitz_estimate=float(np.mean(lipschitz_estimates)) if lipschitz_estimates else 0.0,
    )


def check_physical_plausibility(
    attributions: np.ndarray,
    standard_depths: List[float] = None,
    target_depth_idx: int = 4,
    variable: str = "temp",
) -> Dict[str, float]:
    """
    Oceanography-specific plausibility checks for attributions.
    
    Checks if attributions respect known physical relationships:
    - Surface features should influence thermocline (surface forcing)
    - Deep features should not dominate surface predictions
    - Temporal decay: recent cycles (t-1) should dominate over older (t-3)
    """
    if standard_depths is None:
        standard_depths = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    attr = attributions.squeeze(0)  # (3, 32)
    n_cycles, n_features = attr.shape
    n_depths = n_features // 2  # 16
    
    results = {}
    
    # 1. Temporal decay check (t-1 > t-2 > t-3 for thermocline)
    if variable == "temp":
        cycle_importance = np.sum(np.abs(attr[:, :n_depths]), axis=1)
    else:
        cycle_importance = np.sum(np.abs(attr[:, n_depths:]), axis=1)
    
    # Normalize
    cycle_importance = cycle_importance / (cycle_importance.sum() + 1e-8)
    
    results["temporal_decay_t1_dominates"] = float(cycle_importance[2] > cycle_importance[1] > cycle_importance[0])
    results["t1_weight"] = float(cycle_importance[2])
    results["t2_weight"] = float(cycle_importance[1])
    results["t3_weight"] = float(cycle_importance[0])
    
    # 2. Depth locality (surface features influence surface, deep influence deep)
    # For target depth, check if nearby depths have higher attribution
    if variable == "temp":
        depth_attr = attr[:, :n_depths]  # (3, 16)
    else:
        depth_attr = attr[:, n_depths:]
    
    # Average across cycles
    mean_depth_attr = np.mean(np.abs(depth_attr), axis=0)  # (16,)
    
    # Check if attribution peaks near target depth
    target_attr = mean_depth_attr[target_depth_idx]
    nearby_attr = np.mean(mean_depth_attr[max(0, target_depth_idx-2):target_depth_idx+3])
    far_attr = np.mean(np.concatenate([mean_depth_attr[:max(0, target_depth_idx-2)], 
                                        mean_depth_attr[target_depth_idx+3:]]))
    
    results["depth_locality_ratio"] = float(target_attr / (far_attr + 1e-8))
    results["target_depth_attr"] = float(target_attr)
    results["nearby_depth_attr"] = float(nearby_attr)
    results["far_depth_attr"] = float(far_attr)
    
    # 3. Surface-to-deep coupling (known physics: surface forcing drives thermocline)
    if target_depth_idx >= 2 and target_depth_idx <= 5:  # Thermocline depths
        surface_attr = np.mean(mean_depth_attr[:3])  # 5, 20, 50 dbar
        thermocline_attr = np.mean(mean_depth_attr[2:6])  # 50-150 dbar
        results["surface_thermocline_coupling"] = float(surface_attr / (thermocline_attr + 1e-8))
    
    return results


def run_full_xai_evaluation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    compute_attributions_fn: Callable,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    variable: str = "temp",
    climatology: Optional[torch.Tensor] = None,
    **attribution_kwargs,
) -> Dict:
    """
    Run complete XAI evaluation suite.
    
    Returns dict with all metrics.
    """
    print("Running XAI evaluation suite...")
    
    # Get attributions (raw for evaluation, processed for display)
    from app.ml.xai import compute_integrated_gradients
    raw_attributions, delta = compute_integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        target_variable=target_variable,
        target_depth_idx=target_depth_idx,
    )
    
    # Also get processed results for physical plausibility check
    attr_result = compute_attributions_fn(model, input_tensor, target_variable=target_variable, target_depth_idx=target_depth_idx, **attribution_kwargs)
    
    attributions = raw_attributions
    
    results = {}
    
    # 1. Insertion/Deletion curves
    print("  Computing insertion/deletion curves...")
    ins_result = compute_insertion_deletion_curve(model, input_tensor, attributions, target_variable, climatology=climatology, mode="insertion")
    del_result = compute_insertion_deletion_curve(model, input_tensor, attributions, target_variable, climatology=climatology, mode="deletion")
    results["insertion_auc"] = ins_result.insertion_auc
    results["deletion_auc"] = del_result.deletion_auc
    results["insertion_curve"] = ins_result.insertion_curve.tolist()
    results["deletion_curve"] = del_result.deletion_curve.tolist()
    print(f"    Insertion AUC: {ins_result.insertion_auc:.4f}")
    print(f"    Deletion AUC: {del_result.deletion_auc:.4f}")
    
    # 2. Faithfulness correlation
    print("  Computing faithfulness correlation...")
    faith_result = compute_faithfulness_correlation(model, input_tensor, attributions, target_variable, target_depth_idx)
    results["faithfulness_pearson"] = faith_result.pearson_r
    results["faithfulness_spearman"] = faith_result.spearman_r
    results["faithfulness_p_value"] = faith_result.p_value
    print(f"    Pearson r: {faith_result.pearson_r:.4f} (p={faith_result.p_value:.2e})")
    print(f"    Spearman r: {faith_result.spearman_r:.4f}")
    
    # 3. Stability
    print("  Computing stability metrics...")
    def attr_fn(m, x, **kw):
        # Use compute_integrated_gradients directly for raw attributions
        from app.ml.xai import compute_integrated_gradients
        raw_attr, _ = compute_integrated_gradients(m, x, target_variable=target_variable, target_depth_idx=target_depth_idx)
        return raw_attr
    stab_result = compute_stability_metrics(model, input_tensor, attr_fn, target_variable=target_variable, target_depth_idx=target_depth_idx, **attribution_kwargs)
    results["stability_mean_cos_sim"] = stab_result.mean_cosine_similarity
    results["stability_std_cos_sim"] = stab_result.std_cosine_similarity
    results["local_lipschitz"] = stab_result.local_lipschitz_estimate
    print(f"    Mean cosine sim: {stab_result.mean_cosine_similarity:.4f} ± {stab_result.std_cosine_similarity:.4f}")
    print(f"    Local Lipschitz: {stab_result.local_lipschitz_estimate:.4f}")
    
    # 4. Physical plausibility
    print("  Checking physical plausibility...")
    phys_result = check_physical_plausibility(attributions, target_depth_idx=target_depth_idx, variable=variable)
    results["physical_plausibility"] = phys_result
    print(f"    Temporal decay (t1>t2>t3): {phys_result['temporal_decay_t1_dominates']}")
    print(f"    Depth locality ratio: {phys_result['depth_locality_ratio']:.2f}")
    print(f"    Surface-thermocline coupling: {phys_result.get('surface_thermocline_coupling', 'N/A')}")
    
    return results


def compare_attribution_methods(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_variable: str = "temp",
    target_depth_idx: int = 4,
    n_smoothing_samples: int = 20,
    reference_dataset: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Compare different attribution methods: IG, SmoothGrad, Expected Gradients.
    
    Returns dict with metrics for each method.
    """
    from app.ml.xai import compute_integrated_gradients_with_smoothing, compute_integrated_gradients
    
    print("  Comparing attribution methods...")
    
    methods = {
        "IG": {"smoothing": "none"},
        "SmoothGrad": {"smoothing": "smoothgrad", "n_smoothing_samples": n_smoothing_samples},
        "ExpectedGradients": {"smoothing": "expected_gradients", "n_smoothing_samples": n_smoothing_samples},
    }
    
    results = {}
    
    for method_name, kwargs in methods.items():
        print(f"    Computing {method_name}...")
        try:
            attr, delta = compute_integrated_gradients_with_smoothing(
                model=model,
                input_tensor=input_tensor,
                target_variable=target_variable,
                target_depth_idx=target_depth_idx,
                **kwargs,
            )
            
            # Compute basic metrics
            attr_flat = attr.flatten()
            results[method_name] = {
                "attributions": attr,
                "convergence_delta": delta,
                "mean_abs_attr": float(np.mean(np.abs(attr_flat))),
                "max_abs_attr": float(np.max(np.abs(attr_flat))),
                "attr_std": float(np.std(attr_flat)),
                "sparsity": float((np.abs(attr_flat) < 1e-4).mean()),
            }
            
            print(f"      Delta: {delta:.2e}, Mean|attr|: {results[method_name]['mean_abs_attr']:.4f}")
        except Exception as e:
            print(f"      ERROR: {e}")
            results[method_name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    print("XAI Evaluation module loaded successfully")
    print("Available metrics:")
    print("  - compute_insertion_deletion_curve()")
    print("  - compute_faithfulness_correlation()")
    print("  - compute_stability_metrics()")
    print("  - check_physical_plausibility()")
    print("  - run_full_xai_evaluation()")