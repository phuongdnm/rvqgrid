import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class TimeLSTM(nn.Module):
    """
    Bi-directional LSTM processing along the time axis.
    
    Processes temporal relationships across all time steps.
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
        
        # Reshape: [B*N_q, L, D] - process time axis for each layer independently
        x = x.permute(0, 2, 3, 1)  # [B, N_q, L, D]
        x = x.reshape(B * N_q, L, D)
        
        # LSTM along time
        residual = x
        lstm_out, _ = self.lstm(x)  # [B*N_q, L, hidden*2]
        x = self.proj(lstm_out)  # [B*N_q, L, D]
        x = self.dropout(x)
        x = self.norm(x + residual)
        
        # Reshape back: [B, D, N_q, L]
        x = x.reshape(B, N_q, L, D)
        x = x.permute(0, 3, 1, 2)
        
        return x


class DepthLSTM(nn.Module):
    """
    Bi-directional LSTM processing along the depth (RVQ layer) axis.
    
    Processes hierarchical relationships between RVQ layers.
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
        lstm_out, _ = self.lstm(x)  # [B*L, N_q, hidden*2]
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
        lstm_hidden: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            lstm_hidden: LSTM hidden size
            dropout: Dropout probability
        """
        super().__init__()
        
        # Depth path: Bi-LSTM along N_q axis
        self.depth_path = DepthLSTM(
            d_model=d_model,
            hidden_size=lstm_hidden,
            dropout=dropout,
        )
        
        # Time path: Bi-LSTM along L axis
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


class RVQGridNetEnCodec(nn.Module):
    """
    Grid-based speech separation in EnCodec's RVQ latent space.
    """
    
    def __init__(
        self,
        num_spks: int = 2,
        n_codebooks: int = 16,
        encodec_dim: int = 128,
        grid_dim: int = 128,
        num_blocks: int = 4,
        lstm_hidden: int = 128,
        dropout: float = 0.1,
        activation: str = "elu",
    ):
        """
        Args:
            num_spks: Number of speakers to separate
            n_codebooks: Number of RVQ codebooks
            encodec_dim: EnCodec embedding dimension
            grid_dim: Internal grid dimension
            num_blocks: Number of grid blocks
            lstm_hidden: LSTM hidden size
            dropout: Dropout probability
            activation: Activation function for mask head ("sigmoid", "elu", "relu", "tanh")
        """
        super().__init__()
        self.num_spks = num_spks
        self.n_codebooks = n_codebooks
        self.encodec_dim = encodec_dim
        self.grid_dim = grid_dim
        self.activation_name = activation
        
        self.proj_down = nn.Conv2d(encodec_dim, grid_dim, kernel_size=1)
        
        # Grid blocks
        self.blocks = nn.ModuleList([
            GridBlock(
                d_model=grid_dim,
                lstm_hidden=lstm_hidden,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        
        # EnCodec uses ELU internally, so ELU may work well
        if activation == "sigmoid":
            act_fn = nn.Sigmoid()
        elif activation == "elu":
            act_fn = nn.ELU(alpha=1.0)
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'sigmoid', 'elu', 'relu', or 'tanh'")
        
        # Mask estimation head: grid_dim -> encodec_dim * num_spks
        output_channels = encodec_dim * num_spks
        self.mask_head = nn.Sequential(
            nn.Conv2d(grid_dim, output_channels, kernel_size=1),
            act_fn,
        )
    
    def forward(self, z_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_stack: [B, N_q, D_enc, L] - Stacked RVQ codebook vectors
        
        Returns:
            separated: [num_spks, B, N_q, D_enc, L] - Separated latents
        """
        B, N_q, D, L = z_stack.shape
        
        # Rearrange for Conv2d: [B, D, N_q, L]
        x = z_stack.permute(0, 2, 1, 3)
        
        x = self.proj_down(x)  # [B, grid_dim, N_q, L]
        
        # Grid blocks
        for block in self.blocks:
            x = block(x)  # [B, grid_dim, N_q, L]
        
        # Mask estimation
        masks = self.mask_head(x)  # [B, encodec_dim*num_spks, N_q, L]
        masks = masks.view(B, self.num_spks, D, N_q, L)  # [B, num_spks, D, N_q, L]
        masks = masks.permute(0, 1, 3, 2, 4)  # [B, num_spks, N_q, D, L]
        
        # Apply masks to input
        z_stack_expanded = z_stack.unsqueeze(1)  # [B, 1, N_q, D, L]
        separated = z_stack_expanded * masks  # [B, num_spks, N_q, D, L]
        
        # Rearrange to [num_spks, B, N_q, D, L]
        separated = separated.permute(1, 0, 2, 3, 4)
        
        return separated


def extract_rvq_layers_encodec(
    encodec_wrapper,
    z_e: torch.Tensor,
    n_quantizers: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Extract individual codebook contributions from EnCodec's RVQ.
    
    Args:
        encodec_wrapper: EnCodecWrapperRVQ instance
        z_e: Continuous encoder output [B, 128, L]
        n_quantizers: Number of quantizers to use (None = use wrapper default)
    
    Returns:
        z_q: Quantized sum [B, 128, L]
        codes: Discrete codes [B, N_q, L]
        D_layers: List of individual layer contributions [B, 128, L] each
    """
    z_q, codes, D_layers = encodec_wrapper.get_quantized_features(z_e, n_quantizers)
    return z_q, codes, D_layers


def stack_rvq_layers(D_layers: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack RVQ layer contributions for RVQ-GridNet input.
    
    Args:
        D_layers: List of [B, D, L] tensors
    
    Returns:
        z_stack: [B, N_q, D, L] stacked tensor
    """
    # Stack: [N_q, B, D, L] -> permute to [B, N_q, D, L]
    z_stack = torch.stack(D_layers, dim=0)  # [N_q, B, D, L]
    z_stack = z_stack.permute(1, 0, 2, 3)   # [B, N_q, D, L]
    return z_stack


def decode_separated_encodec(
    encodec_wrapper,
    separated: torch.Tensor,
    original_length: int,
) -> torch.Tensor:
    """
    Decode separated RVQ vectors back to waveforms.
    
    Args:
        encodec_wrapper: EnCodecWrapperRVQ instance
        separated: [num_spks, B, N_q, D, L] - Separated latents
        original_length: Original audio length before resampling
    
    Returns:
        waveforms: [num_spks, B, 1, T] - Separated audio
    """
    num_spks, B, N_q, D, L = separated.shape
    waveforms = []
    
    for spk in range(num_spks):
        # Sum over codebooks: [B, N_q, D, L] -> [B, D, L]
        z_spk = separated[spk].sum(dim=1)  # [B, 128, L]
        
        # Decode using EnCodec decoder
        wav = encodec_wrapper.get_decoded_signal(z_spk, original_length)
        waveforms.append(wav)
    
    return torch.stack(waveforms, dim=0)  # [num_spks, B, 1, T]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

