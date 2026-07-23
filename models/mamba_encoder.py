import torch
import torch.nn as nn


try:
    from mamba_ssm import Mamba as OfficialMamba
    from mamba_ssm import __version__ as MAMBA_SSM_VERSION

    _MAMBA_IMPORT_ERROR = None
except Exception as exc:
    OfficialMamba = None
    MAMBA_SSM_VERSION = None
    _MAMBA_IMPORT_ERROR = exc


class MambaTemporalEncoder(nn.Module):
    """Pool a sequence after processing it with the official Mamba block."""

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 1,
    ):
        super().__init__()
        if OfficialMamba is None:
            raise ImportError(
                "The official state-spaces/mamba package is required. Install "
                "`mamba-ssm` in its supported Linux CUDA environment."
            ) from _MAMBA_IMPORT_ERROR
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError("Mamba dimensions must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [
                OfficialMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, temporal_tokens: torch.Tensor) -> torch.Tensor:
        if temporal_tokens.dim() != 3:
            raise ValueError(
                "temporal_tokens must have shape [batch, sequence, d_model]"
            )
        if temporal_tokens.size(1) == 0:
            raise ValueError("temporal sequence must contain at least one token")
        if temporal_tokens.size(-1) != self.d_model:
            raise ValueError(
                f"Expected token dimension {self.d_model}, "
                f"got {temporal_tokens.size(-1)}"
            )

        hidden_states = temporal_tokens
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Mamba is causal, so the final hidden state summarizes z_1 ... z_t.
        return hidden_states[:, -1, :]
