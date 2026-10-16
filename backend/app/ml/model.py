"""
Physics-Informed Bi-LSTM PyTorch Model

2-layer Bi-LSTM with dropout (p=0.2) and dual linear projection heads
for Temperature and Salinity. Automatic GPU (cuda/mps) device binding.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PhysicsInformedBiLSTM(nn.Module):
    """
    Physics-Informed Bidirectional LSTM for Argo profile forecasting.
    
    Architecture:
    - Input: (batch, seq_len=3, features=32) where 32 = 16 T + 16 S
    - 2-layer Bi-LSTM: 128 hidden units per direction (256 total)
    - Dropout: p=0.2 between layers and before output heads
    - Dual output heads: Temperature (16) and Salinity (16)
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 16,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Dropout before output heads
        self.dropout = nn.Dropout(dropout)
        
        # Dual output heads
        self.temp_head = nn.Linear(lstm_out_dim, output_dim)
        self.sal_head = nn.Linear(lstm_out_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (standard LSTM practice)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # Output heads
        nn.init.xavier_uniform_(self.temp_head.weight)
        nn.init.xavier_uniform_(self.sal_head.weight)
        self.temp_head.bias.data.fill_(0)
        self.sal_head.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len=3, input_dim=32)
               where 32 = 16 temperature + 16 salinity features
        
        Returns:
            Tuple of (temp_pred, sal_pred) each of shape (batch, output_dim=16)
        """
        # LSTM forward pass
        # Output shape: (batch, seq_len, hidden_dim * 2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last time step output
        # Shape: (batch, hidden_dim * 2)
        last_out = lstm_out[:, -1, :]
        
        # Apply dropout
        last_out = self.dropout(last_out)
        
        # Dual heads
        temp_pred = self.temp_head(last_out)  # (batch, 16)
        sal_pred = self.sal_head(last_out)    # (batch, 16)
        
        return temp_pred, sal_pred
    
    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device


def get_device() -> torch.device:
    """
    Get the optimal compute device with priority: cuda > mps > cpu.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[ArgoCast ML] Using CUDA: {torch.cuda.get_device_name(0)}")
        # Enable cuDNN benchmark for optimized convolutions
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[ArgoCast ML] Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[ArgoCast ML] Using CPU (fallback)")
    return device


def create_model(device: Optional[torch.device] = None) -> PhysicsInformedBiLSTM:
    """
    Factory function to create and move model to device.
    """
    if device is None:
        device = get_device()
    
    model = PhysicsInformedBiLSTM()
    model = model.to(device)
    return model


# Single-output wrapper for Captum Integrated Gradients
class SingleOutputModelWrapper(nn.Module):
    """
    Wraps the multi-head Bi-LSTM to output a single scalar for Captum attribution.
    
    This is required because Captum's IntegratedGradients only works with
    single-output models. We select one variable (temp/sal) and one depth index.
    """
    
    def __init__(
        self,
        base_model: PhysicsInformedBiLSTM,
        target_variable: str = "temp",
        target_depth_idx: int = 4  # 100 dbar (thermocline core)
    ):
        super().__init__()
        self.base_model = base_model
        self.target_variable = target_variable
        self.target_depth_idx = target_depth_idx
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning single scalar output.
        
        Args:
            x: Input tensor (batch, seq_len, features)
        
        Returns:
            Scalar tensor for the target variable at target depth
        """
        temp_pred, sal_pred = self.base_model(x)
        
        if self.target_variable == "temp":
            return temp_pred[:, self.target_depth_idx].unsqueeze(1)
        elif self.target_variable == "sal":
            return sal_pred[:, self.target_depth_idx].unsqueeze(1)
        else:
            raise ValueError(f"Unknown target_variable: {self.target_variable}")


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing PhysicsInformedBiLSTM...")
    
    device = get_device()
    model = create_model(device)
    
    # Test input: batch=32, seq_len=3, features=32
    x = torch.randn(32, 3, 32, device=device)
    
    temp_pred, sal_pred = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Temp output shape: {temp_pred.shape}")
    print(f"Sal output shape: {sal_pred.shape}")
    
    # Verify shapes
    assert temp_pred.shape == (32, 16), f"Expected (32, 16), got {temp_pred.shape}"
    assert sal_pred.shape == (32, 16), f"Expected (32, 16), got {sal_pred.shape}"
    
    print("OK Model forward pass successful!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")