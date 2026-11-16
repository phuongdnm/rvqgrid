import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class TimeLSTM(nn.Module):
    """
    Bi-directional LSTM processing along the time axis.
    """
    
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            hidden_size: LSTM hidden size
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # Project back to d_model (2*hidden due to bidirectional)
        self.proj = nn.Linear(hidden_size * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, N_q, L] - Input tensor
        Returns:
            x: [B, D, N_q, L] - Output tensor
        """
        B, D, N_q, L = x.shape
        
        # Reshape: [B*N_q, L, D]
        x = x.permute(0, 2, 3, 1)  # [B, N_q, L, D]
        x = x.reshape(B * N_q, L, D)
        
        # LSTM
        residual = x
        lstm_out, _ = self.lstm(x)  # [B*N_q, L, hidden*2]
        x = self.proj(lstm_out)  # [B*N_q, L, D]
        x = self.dropout(x)  # Apply dropout after projection
        x = self.norm(x + residual)
        
        # Reshape back: [B, D, N_q, L]
        x = x.reshape(B, N_q, L, D)
        x = x.permute(0, 3, 1, 2)
        
        return x


class DepthLSTM(nn.Module):
    """
    Bi-directional LSTM processing along the depth (RVQ layer) axis.
    """
    
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            hidden_size: LSTM hidden size
            dropout: Dropout probability
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_size * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, N_q, L] - Input tensor
        Returns:
            x: [B, D, N_q, L] - Output tensor
        """
        B, D, N_q, L = x.shape
        
        # Reshape: [B*L, N_q, D]
        x = x.permute(0, 3, 2, 1).reshape(B * L, N_q, D)
        
        residual = x
        lstm_out, _ = self.lstm(x)
        x = self.proj(lstm_out)
        x = self.dropout(x)
        x = self.norm(x + residual)
        
        # Reshape back: [B, D, N_q, L]
        x = x.reshape(B, L, N_q, D).permute(0, 3, 2, 1)
        
        return x


class GridBlock(nn.Module):
    """
    Single Grid block with Depth (LSTM) and Time (LSTM) paths.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        lstm_hidden: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension (e.g., 256)
            n_layers: Number of RVQ layers (e.g., 12)
            lstm_hidden: LSTM hidden size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.depth_path = DepthLSTM(
            d_model=d_model,
            hidden_size=lstm_hidden,
            dropout=dropout,
        )
        
        # Time path: Bi-LSTM
        self.time_path = TimeLSTM(
            d_model=d_model,
            hidden_size=lstm_hidden,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, N_q, L] - Input tensor
        Returns:
            x: [B, D, N_q, L] - Output tensor
        """
        # Depth path (process along N_q dimension)
        x = self.depth_path(x)
        
        # Time path (process along L dimension)
        x = self.time_path(x)
        
        return x


class RVQGridNet(nn.Module):
    """
    Grid-based speech separation in RVQ latent space.
    
    This model treats the RVQ Depth (Number of Codebooks) as a structural dimension
    and applies GridNet-style processing across both Time and Quantization Depth.
    """
    
    def __init__(
        self,
        num_spks: int = 2,
        n_codebooks: int = 12,
        dac_dim: int = 1024,
        grid_dim: int = 256,
        num_blocks: int = 4,
        lstm_hidden: int = 256,
        mask_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_spks: Number of speakers to separate (default: 2)
            n_codebooks: Number of RVQ codebooks (default: 12 for DAC 16kHz)
            dac_dim: DAC embedding dimension (default: 1024)
            grid_dim: Internal grid dimension (default: 256)
            num_blocks: Number of grid blocks (default: 4)
            lstm_hidden: LSTM hidden size (default: 256)
            mask_hidden_dim: Hidden dim for mask head (0 for linear, >0 for MLP)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()
        self.num_spks = num_spks
        self.n_codebooks = n_codebooks
        self.dac_dim = dac_dim
        self.grid_dim = grid_dim
        
        self.proj_down = nn.Conv2d(dac_dim, grid_dim, kernel_size=1)
        
        # Grid blocks
        self.blocks = nn.ModuleList([
            GridBlock(
                d_model=grid_dim,
                n_layers=n_codebooks,
                lstm_hidden=lstm_hidden,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        
        # Mask estimation head: grid_dim -> dac_dim * num_spks
        self.mask_hidden_dim = mask_hidden_dim
        if mask_hidden_dim > 0:
            # MLP with non-linear activation
            self.mask_head = nn.Sequential(
                nn.Conv2d(grid_dim, mask_hidden_dim, kernel_size=1),  # 256 → 512
                nn.ReLU(),
                nn.Conv2d(mask_hidden_dim, dac_dim * num_spks, kernel_size=1),  # 512 → 2048
                nn.Sigmoid(),  # Mask bounded [0, 1]
            )
        else:
            # Simple mode: Linear projection only
            self.mask_head = nn.Sequential(
                nn.Conv2d(grid_dim, dac_dim * num_spks, kernel_size=1),  # 256 → 2048
                nn.Sigmoid(),
            )
    
    def forward(self, z_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_stack: [B, N_q, D_dac, L] - Stacked RVQ codebook vectors
        
        Returns:
            separated: [num_spks, B, N_q, D_dac, L] - Separated latents
        """
        B, N_q, D, L = z_stack.shape
        
        # Rearrange for Conv2d: [B, D, N_q, L]
        x = z_stack.permute(0, 2, 1, 3)
        
        # Bottleneck projection
        x = self.proj_down(x)  # [B, grid_dim, N_q, L]
        
        # Grid blocks
        for block in self.blocks:
            x = block(x)  # [B, grid_dim, N_q, L]
        
        # Mask estimation
        masks = self.mask_head(x)  # [B, dac_dim*num_spks, N_q, L]
        masks = masks.view(B, self.num_spks, D, N_q, L)  # [B, num_spks, D, N_q, L]
        masks = masks.permute(0, 1, 3, 2, 4)  # [B, num_spks, N_q, D, L]
        
        # Apply masks to input
        z_stack_expanded = z_stack.unsqueeze(1)  # [B, 1, N_q, D, L]
        separated = z_stack_expanded * masks  # [B, num_spks, N_q, D, L]
        
        # Rearrange to [num_spks, B, N_q, D, L]
        separated = separated.permute(1, 0, 2, 3, 4)
        
        return separated


def extract_rvq_layers(dac_model, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract individual codebook contributions from RVQ.
    
    Args:
        dac_model: DACWrapper instance or DAC model with quantizer
        z_e: Continuous encoder output [B, 1024, L]
    
    Returns:
        z_stack: Stacked codebook vectors [B, N_q, 1024, L]
        codes: Discrete codes [B, N_q, L]
    """
    if hasattr(dac_model, 'model'):
        model = dac_model.model
    else:
        model = dac_model
    
    # Get quantized output and codes
    z_q, codes, latents, _, _ = model.quantizer(z_e, n_quantizers=None)
    
    # Extract individual D_i from codes
    D_layers = []
    for i, quantizer_layer in enumerate(model.quantizer.quantizers):
        indices = codes[:, i, :]  # [B, L]
        vectors = quantizer_layer.codebook(indices)  # [B, L, 8]
        vectors = vectors.permute(0, 2, 1)  # [B, 8, L]
        D_i = quantizer_layer.out_proj(vectors)  # [B, 1024, L]
        D_layers.append(D_i)
    
    # Stack: [B, N_q, 1024, L]
    z_stack = torch.stack(D_layers, dim=1)
    
    return z_stack, codes


def decode_separated(dac_model, separated: torch.Tensor, original_length: int) -> torch.Tensor:
    """
    Decode separated RVQ vectors back to waveforms.

    Args:
        dac_model: DACWrapper instance
        separated: [num_spks, B, N_q, D, L] - Separated latents
        original_length: Original audio length before resampling

    Returns:
        waveforms: [num_spks, B, 1, T] - Separated audio
    """
    num_spks, B, N_q, D, L = separated.shape
    waveforms = []

    for spk in range(num_spks):
        # Sum over codebooks: [B, N_q, D, L] -> [B, D, L]
        z_spk = separated[spk].sum(dim=1)  # [B, 1024, L]

        # Decode using DAC decoder
        wav = dac_model.get_decoded_signal(z_spk, original_length)
        waveforms.append(wav)
    
    return torch.stack(waveforms, dim=0)  # [num_spks, B, 1, T]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

