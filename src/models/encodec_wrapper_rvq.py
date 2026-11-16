import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Optional, Tuple, List

from encodec import EncodecModel

class EnCodecWrapperRVQ(nn.Module):
    """
    Wrapper model for EnCodec with RVQ layer extraction support.
    
    EnCodec 24kHz bandwidth to codebook mapping:
        - 1.5 kbps - 2 codebooks
        - 3.0 kbps - 4 codebooks
        - 6.0 kbps - 8 codebooks
        - 12.0 kbps - 16 codebooks
        - 24.0 kbps - 32 codebooks
    """
    BANDWIDTH_TO_CODEBOOKS = {
        1.5: 2,
        3.0: 4,
        6.0: 8,
        12.0: 16,
        24.0: 32,
    }
    
    def __init__(
        self,
        input_sample_rate: int = 8000,
        encodec_sample_rate: int = 24000,
        bandwidth: float = 12.0,
        Freeze: bool = True,
    ):
        """
        Args:
            input_sample_rate: Input audio sample rate (default: 8000)
            encodec_sample_rate: EnCodec model's native sample rate (24000)
            bandwidth: Target bandwidth in kbps (1.5, 3.0, 6.0, 12.0, or 24.0)
            Freeze: If True, freeze EnCodec parameters
        """
        super(EnCodecWrapperRVQ, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.encodec_sample_rate = encodec_sample_rate
        self.bandwidth = bandwidth
        
        self.n_codebooks = self.BANDWIDTH_TO_CODEBOOKS.get(bandwidth, 16)
        
        self.embedding_dim = 128
        
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        
        self.to_encodec_sampler = T.Resample(input_sample_rate, encodec_sample_rate)
        self.to_input_sampler = T.Resample(encodec_sample_rate, input_sample_rate)

        def count_all_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'EnCodec Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'EnCodec Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')
            print(f'EnCodec bandwidth: {bandwidth} kbps ({self.n_codebooks} codebooks)')
            print(f'EnCodec embedding dimension: {self.embedding_dim}')
        else:
            print(f'EnCodec Model with {count_parameters(self.model)/1000000:.2f}M trainable parameters loaded')
        

    def resample_audio(
        self, 
        x: torch.Tensor, 
        condition: str, 
        normalize: bool = False
    ) -> torch.Tensor:
        """
        Resample audio between input rate and EnCodec rate.
        
        Args:
            x: Audio tensor [B, C, T]
            condition: "encodec" to upsample, "input" to downsample
            normalize: If True, normalize audio to prevent clipping
        
        Returns:
            Resampled audio tensor
        """
        device = x.device

        assert len(x.shape) == 3, f"Expected 3D tensor [B, C, T], got {x.shape}"
        B, C, T = x.shape
        assert C == 1, "Model should only handle single channel"

        if condition == "encodec":
            x_resamp = self.to_encodec_sampler(x)
        elif condition == "input":
            x_resamp = self.to_input_sampler(x)
        else:
            raise ValueError(f"Unknown condition: {condition}. Use 'encodec' or 'input'")
        
        if normalize:
            x_resamp = x_resamp / torch.max(x_resamp.abs(), dim=2, keepdim=True)[0].clamp(min=1e-8)

        return x_resamp.to(device)

    def get_encoded_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Encode audio to continuous latent features.
        
        Args:
            x: Audio tensor [B, 1, T] at input_sample_rate
        
        Returns:
            x_enc: Encoded features [B, 128, L]
            original_length: Original audio length for reconstruction
        """
        original_length = x.shape[-1]

        x = self.resample_audio(x, "encodec")
        
        x_enc = self.model.encoder(x).detach()

        return x_enc, original_length
    
    def get_quantized_features(
        self, 
        x: torch.Tensor, 
        n_quantizers: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Quantize continuous features using RVQ.
        
        Args:
            x: Continuous features [B, 128, L]
            n_quantizers: Number of quantizers to use (None = use bandwidth setting)
        
        Returns:
            z_q: Quantized features (sum of all layers) [B, 128, L]
            codes: Discrete codes [B, n_q, L]
            D_layers: List of individual layer contributions [B, 128, L] each
        """
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        
        vq_layers = self.model.quantizer.vq.layers
        
        B, D, L = x.shape
        residual = x.clone()
        codes_list = []
        D_layers = []
        
        for i in range(n_quantizers):
            layer = vq_layers[i]
            
            # Encode residual to get codes for this layer
            indices = layer.encode(residual)  # [B, L]
            codes_list.append(indices)
            
            # Decode codes to get quantized contribution
            D_i = layer.decode(indices)  # [B, D, L]
            D_layers.append(D_i)
            
            # Update residual
            residual = residual - D_i
        
        # Stack codes: [B, n_q, L]
        codes = torch.stack(codes_list, dim=1)
        
        # Sum of all D_i gives quantized output
        z_q = sum(D_layers)
        
        return z_q, codes, D_layers

    def _extract_rvq_layers(
        self, 
        codes: torch.Tensor, 
        n_quantizers: int
    ) -> List[torch.Tensor]:
        """
        Extract individual RVQ layer contributions from codes.
        
        Args:
            codes: Discrete codes [B, K, L]
            n_quantizers: Number of quantizers to extract
        
        Returns:
            D_layers: List of layer contributions, each [B, 128, L]
        """
        vq_layers = self.model.quantizer.vq.layers
        D_layers = []
        
        for i in range(n_quantizers):
            layer = vq_layers[i]
            indices = codes[:, i, :]  # [B, L]
            
            D_i = layer.decode(indices)  # [B, D, L]
            D_layers.append(D_i)
        
        return D_layers

    def get_rvq_layers_from_codes(
        self, 
        codes: torch.Tensor, 
        n_quantizers: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Extract individual RVQ layer contributions from pre-computed codes.
        
        Args:
            codes: Discrete codes [B, K, L]
            n_quantizers: Number of layers to extract (None = all available)
        
        Returns:
            D_layers: List of layer contributions, each [B, 128, L]
        """
        if n_quantizers is None:
            n_quantizers = codes.shape[1]
        
        return self._extract_rvq_layers(codes, n_quantizers)

    def get_sum_from_layers(self, D_layers: List[torch.Tensor]) -> torch.Tensor:
        """
        Sum individual RVQ layer contributions.
        
        Args:
            D_layers: List of layer contributions [B, 128, L] each
        
        Returns:
            z_sum: Summed features [B, 128, L]
        """
        return sum(D_layers)

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode codes back to quantized features using quantizer.decode().
        
        Note: EnCodec expects codes in [K, B, L] format for decode
        
        Args:
            codes: Discrete codes [B, K, L] (from encode)
        
        Returns:
            z_q: Quantized features [B, 128, L]
        """
        # Transpose to [K, B, L] for EnCodec's decode
        codes_transposed = codes.transpose(0, 1)  # [K, B, L]
        z_q = self.model.quantizer.decode(codes_transposed)  # [B, D, L]
        return z_q

    def get_decoded_signal(
        self, 
        x: torch.Tensor, 
        original_length: int
    ) -> torch.Tensor:
        """
        Decode latent features to audio.
        
        Args:
            x: Latent features [B, 128, L]
            original_length: Target audio length
        
        Returns:
            out: Decoded audio [B, 1, T]
        """
        if torch.is_grad_enabled():
            self.model.train()
        
        y_hat = self.model.decoder(x)
        out = self.resample_audio(y_hat, "input")
        
        # Length matching
        if out.shape[-1] != original_length:
            T_origin = original_length
            T_est = out.shape[-1]
            
            if T_origin > T_est:
                out = F.pad(out, (0, T_origin - T_est))
            else:
                out = out[:, :, :T_origin]
        return out

    def encode_decode(
        self, 
        x: torch.Tensor, 
        n_quantizers: Optional[int] = None
    ) -> torch.Tensor:
        """
        Full encode-quantize-decode pipeline (for testing reconstruction).
        
        Args:
            x: Audio tensor [B, 1, T] at input_sample_rate
            n_quantizers: Number of quantizers (None = use bandwidth setting)
        
        Returns:
            reconstructed: Reconstructed audio [B, 1, T]
        """
        original_length = x.shape[-1]
        
        # Encode
        x_enc, _ = self.get_encoded_features(x)
        
        # Quantize
        z_q, codes, D_layers = self.get_quantized_features(x_enc, n_quantizers)
        
        # Decode
        out = self.get_decoded_signal(z_q, original_length)
        
        return out


EnCodecWrapper = EnCodecWrapperRVQ
