import torch
import torch.nn.functional as F
import torchaudio.transforms as T

try:
    import dac
except ImportError:
    raise ImportError("Please install descript-audio-codec: pip install descript-audio-codec")


class DACWrapperRVQ:
    """
    Wrapper model for Descript Audio Codec.
    """

    def __init__(
        self,
        input_sample_rate: int = 8000,
        DAC_model_path: str = None,
        DAC_sample_rate: int = 16000,
        Freeze: bool = True,
    ):
        """
        Args:
            input_sample_rate: Input audio sample rate
            DAC_model_path: Path to DAC checkpoint
            DAC_sample_rate: DAC model's native sample rate
            Freeze: If True, freeze DAC parameters
        """
        super(DACWrapperRVQ, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.DAC_sample_rate = DAC_sample_rate
        self.dac_model_type = "24khz" if DAC_sample_rate == 24000 else "16khz"
        
        if DAC_model_path is None:
            model_path = dac.utils.download(model_type=self.dac_model_type)
            print(f"Used pre-trained DAC model for {self.dac_model_type}")
        else:
            model_path = DAC_model_path
        
        self.model = dac.DAC.load(model_path)

        self.dac_sampler = T.Resample(input_sample_rate, DAC_sample_rate)
        self.org_sampler = T.Resample(DAC_sample_rate, input_sample_rate)

        def count_all_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'DAC Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'DAC Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')
        else:
            print(f'DAC Model with {count_parameters(self.model)/1000000:.2f}M trainable parameters loaded')

    def resample_audio(self, x, condition, normalize=False):
        """
        Resample audio between input rate and DAC rate.
        
        Args:
            x: Audio tensor [B, C, T]
            condition: "dac" to upsample, "org" to downsample
            normalize: If True, normalize audio to prevent clipping (default: False)
        
        Returns:
            Resampled audio tensor
        """
        device = x.device

        assert len(x.shape) == 3
        B, C, T = x.shape
        assert C == 1, "Model should only handle single channel"

        if condition == "dac":
            x_resamp = self.dac_sampler(x)
        elif condition == "org":
            x_resamp = self.org_sampler(x)
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Only normalize if explicitly requested
        if normalize:
            x_resamp = x_resamp / torch.max(x_resamp.abs(), dim=2, keepdim=True)[0].clamp(min=1e-8)

        return x_resamp.to(device)

    def get_encoded_features(self, x):
        """
        Encode audio to continuous latent features.
        
        Args:
            x: Audio tensor [B, 1, T] at input_sample_rate
        
        Returns:
            x_enc: Encoded features [B, 1024, L]
            original_length: Original audio length for reconstruction
        """
        original_length = x.shape[-1]

        x = self.resample_audio(x, "dac")
        x = self.model.preprocess(x, self.DAC_sample_rate)
        x_enc = self.model.encoder(x)

        return x_enc, original_length
    
    def get_quantized_features(self, x, n_quantizers=None):
        """
        Quantize continuous features using RVQ.
        
        Args:
            x: Continuous features [B, 1024, L]
            n_quantizers: Number of quantizers to use (None = all)
        
        Returns:
            x_qnt: Quantized features [B, 1024, L]
            codes: Discrete codes [B, N_q, L]
            latents: Intermediate latents
            commitment_loss: VQ commitment loss
            codebook_loss: VQ codebook loss
        """
        x_qnt, codes, latents, commitment_loss, codebook_loss = self.model.quantizer(x, n_quantizers)
        return x_qnt, codes, latents, commitment_loss, codebook_loss

    def get_decoded_signal(self, x, original_length):
        """
        Decode latent features to audio.

        Args:
            x: Latent features [B, 1024, L]
            original_length: Target audio length
        
        Returns:
            out: Decoded audio [B, 1, T]
        """
        y_hat = self.model.decoder(x)
        out = self.resample_audio(y_hat, "org")
        
        # Length matching (pad/truncate at end)
        if out.shape[-1] != original_length:
            T_origin = original_length
            T_est = out.shape[-1]
            
            if T_origin > T_est:
                out = F.pad(out, (0, T_origin - T_est))
            else:
                out = out[:, :, :T_origin]
        return out



DACWrapper = DACWrapperRVQ
