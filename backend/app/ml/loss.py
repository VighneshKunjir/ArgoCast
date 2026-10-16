"""
Physics-Constrained Loss Function for ArgoCast Bi-LSTM

Implements composite loss:
L = MSE(T) + 2.5*MSE(S) + 10.0*L_stability + 1.5*L_therm

Where:
- L_stability penalizes density inversions (∂σ_θ/∂z < 0)
- L_therm penalizes thermocline gradient errors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class PhysicsConstrainedLoss(nn.Module):
    """
    Physics-constrained loss function combining MSE with TEOS-10 physics penalties.
    """
    
    def __init__(
        self,
        lambda_sal: float = 2.5,
        lambda_stability: float = 10.0,
        lambda_therm: float = 1.5,
        standard_depths: Optional[torch.Tensor] = None,
        temp_mean: Optional[torch.Tensor] = None,
        temp_std: Optional[torch.Tensor] = None,
        sal_mean: Optional[torch.Tensor] = None,
        sal_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.lambda_sal = lambda_sal
        self.lambda_stability = lambda_stability
        self.lambda_therm = lambda_therm

        # Standard depths for gradient computation
        if standard_depths is None:
            self.standard_depths = torch.tensor(
                [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
                dtype=torch.float32
            )
        else:
            self.standard_depths = standard_depths

        # Register as buffer so it moves with model.to(device)
        self.register_buffer('depths', self.standard_depths)

        # StandardScaler stats (per-depth) for denormalizing predictions back
        # to physical units BEFORE the TEOS-10 polynomial, which is only
        # valid in physical units. None -> inputs already physical.
        def _buf(t):
            return torch.as_tensor(t, dtype=torch.float32) if t is not None else None

        for name, tensor in (("temp_mean", _buf(temp_mean)), ("temp_std", _buf(temp_std)),
                             ("sal_mean", _buf(sal_mean)), ("sal_std", _buf(sal_std))):
            if tensor is None:
                tensor = torch.zeros(16) if "mean" in name else torch.ones(16)
            self.register_buffer(name, tensor)
    
    def compute_potential_density(self, temp: torch.Tensor, sal: torch.Tensor) -> torch.Tensor:
        """
        Compute potential density anomaly sigma_theta (differentiable).

        Uses the pure NumPy/PyTorch UNESCO/TEOS-10 polynomial approximation
        so gradients flow during backprop (gsw C-extension
        calls are not differentiable and must not be used in the loss graph):
        sigma_theta ~= 28.14 - 0.0735*T - 0.00469*T^2 + 0.802*(S - 35.0)

        Args:
            temp: Temperature tensor (..., 16)
            sal: Salinity tensor (..., 16)

        Returns:
            Potential density anomaly tensor (..., 16)
        """
        # Denormalize to physical units: the polynomial is calibrated to
        # physical T/S, not standardized model outputs.
        temp = temp * self.temp_std + self.temp_mean
        sal = sal * self.sal_std + self.sal_mean
        sigma = (28.14
                 - 0.0735 * temp
                 - 0.00469 * temp ** 2
                 + 0.802 * (sal - 35.0))

        return sigma
    
    def compute_stability_loss(self, temp: torch.Tensor, sal: torch.Tensor) -> torch.Tensor:
        """
        Compute stability loss penalizing density inversions.
        
        L_stability = Σ max(0, -∂σ_θ/∂z) over all depth intervals
        
        Where ∂σ_θ/∂z = (σ[i+1] - σ[i]) / (z[i+1] - z[i])
        
        Args:
            temp: Predicted temperature (batch, 16)
            sal: Predicted salinity (batch, 16)
        
        Returns:
            Scalar loss value
        """
        sigma = self.compute_potential_density(temp, sal)  # (batch, 16)
        
        # Compute vertical gradient ∂σ/∂z
        # Using finite differences: (σ[i+1] - σ[i]) / (z[i+1] - z[i])
        d_sigma = sigma[:, 1:] - sigma[:, :-1]  # (batch, 15)
        d_depth = self.depths[1:] - self.depths[:-1]  # (15,)
        
        # Stability gradient
        stability = d_sigma / d_depth  # (batch, 15)
        
        # Penalty: max(0, -stability) for each depth interval
        # This penalizes when density decreases with depth
        violations = torch.relu(-stability)  # (batch, 15)
        
        # Sum over depth intervals and average over batch
        loss = violations.sum(dim=1).mean()
        
        return loss
    
    def compute_thermocline_loss(self, temp_pred: torch.Tensor, temp_true: torch.Tensor) -> torch.Tensor:
        """
        Compute thermocline gradient loss.
        
        L_therm = |(T_75 - T_150)/75 - (T̂_75 - T̂_150)/75|^2
        
        Depth indices: 75 dbar = index 3, 150 dbar = index 5
        
        Args:
            temp_pred: Predicted temperature (batch, 16)
            temp_true: True temperature (batch, 16)
        
        Returns:
            Scalar loss value
        """
        # Thermocline core: 75-150 dbar (indices 3 and 5), in physical units
        pred_phys = temp_pred * self.temp_std + self.temp_mean
        true_phys = temp_true * self.temp_std + self.temp_mean
        pred_grad = (pred_phys[:, 3] - pred_phys[:, 5]) / (self.depths[5] - self.depths[3])
        true_grad = (true_phys[:, 3] - true_phys[:, 5]) / (self.depths[5] - self.depths[3])

        loss = F.mse_loss(pred_grad, true_grad)
        
        return loss
    
    def forward(
        self,
        temp_pred: torch.Tensor,
        temp_true: torch.Tensor,
        sal_pred: torch.Tensor,
        sal_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-constrained loss.
        
        Args:
            temp_pred: Predicted temperature (batch, 16)
            temp_true: True temperature (batch, 16)
            sal_pred: Predicted salinity (batch, 16)
            sal_true: True salinity (batch, 16)
        
        Returns:
            (total_loss, loss_components_dict)
        """
        # MSE losses
        mse_temp = F.mse_loss(temp_pred, temp_true)
        mse_sal = F.mse_loss(sal_pred, sal_true)
        
        # Physics losses
        stability_loss = self.compute_stability_loss(temp_pred, sal_pred)
        thermocline_loss = self.compute_thermocline_loss(temp_pred, temp_true)
        
        # Weighted total loss
        total_loss = (
            mse_temp 
            + self.lambda_sal * mse_sal 
            + self.lambda_stability * stability_loss 
            + self.lambda_therm * thermocline_loss
        )
        
        # Return components for logging
        components = {
            'total': total_loss.item(),
            'mse_temp': mse_temp.item(),
            'mse_sal': mse_sal.item(),
            'stability': stability_loss.item(),
            'thermocline': thermocline_loss.item()
        }
        
        return total_loss, components


def test_stability_loss():
    """
    Self-check: backprop through a deliberate density
    inversion produces L_stability > 0.
    """
    print("Testing Physics-Constrained Loss...")
    torch.manual_seed(42)

    loss_fn = PhysicsConstrainedLoss()

    batch_size = 4

    # Stable profile: T decreases with depth, S increases slightly
    temp_normal = torch.tensor([
        [28.0, 27.5, 27.0, 25.0, 23.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.5]
    ] * batch_size, dtype=torch.float32)
    sal_normal = torch.tensor([
        [34.5, 34.6, 34.7, 34.8, 34.9, 35.0, 35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.9, 36.0]
    ] * batch_size, dtype=torch.float32)

    # Deliberate inversion: warmer water placed deeper (unphysical)
    temp_inverted = temp_normal.clone().detach().requires_grad_(True)
    sal_inverted = sal_normal.clone().detach().requires_grad_(True)
    with torch.no_grad():
        temp_inverted[:, 10:14] += 5.0

    # Stable case must have zero penalty
    _, comp_normal = loss_fn(temp_normal, temp_normal, sal_normal, sal_normal)
    print(f"Normal profile - Stability loss: {comp_normal['stability']:.6f}")

    # Inverted case: forward + full backprop
    loss_inverted, comp_inverted = loss_fn(
        temp_inverted, temp_inverted.detach(), sal_inverted, sal_inverted.detach()
    )
    print(f"Inverted profile - Stability loss: {comp_inverted['stability']:.6f}")
    assert comp_inverted['stability'] > 0.0, "L_stability must be > 0 for inversion"

    loss_inverted.backward()
    assert temp_inverted.grad is not None, "No gradient flowed to temperature"
    assert torch.isfinite(temp_inverted.grad).all(), "Non-finite gradients"
    assert temp_inverted.grad.abs().sum().item() > 0.0, "Zero gradients on inversion"

    print("OK Stability loss correctly penalizes inversions with backprop")
    print(f"  Normal: {comp_normal['stability']:.6f}, Inverted: {comp_inverted['stability']:.6f}")
    print(f"  Grad norm: {temp_inverted.grad.abs().sum().item():.6f}")

    # Full loss smoke test
    total_loss, comps = loss_fn(temp_normal, temp_normal, sal_normal, sal_normal)
    print("OK Total loss components:", comps)

    return True


if __name__ == "__main__":
    test_stability_loss()