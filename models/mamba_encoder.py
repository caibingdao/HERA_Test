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

    def _forward_blocks(self, temporal_tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = temporal_tokens
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states

    def forward(
        self,
        temporal_tokens: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
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

        if padding_mask is None:
            return self._forward_blocks(temporal_tokens)[:, -1, :]

        if padding_mask.shape != temporal_tokens.shape[:2]:
            raise ValueError(
                "padding_mask must have shape [batch, sequence]"
            )
        padding_mask = padding_mask.bool()
        padding_counts = padding_mask.sum(dim=1)
        expected_padding = (
            torch.arange(
                temporal_tokens.size(1),
                device=temporal_tokens.device,
            ).unsqueeze(0)
            < padding_counts.unsqueeze(1)
        )
        if not torch.equal(padding_mask, expected_padding):
            raise ValueError("Only left-aligned temporal padding is supported")

        valid_lengths = temporal_tokens.size(1) - padding_counts
        pooled_rows = [None] * temporal_tokens.size(0)
        for valid_length in torch.unique(valid_lengths).tolist():
            row_indices = torch.nonzero(
                valid_lengths == valid_length,
                as_tuple=False,
            ).squeeze(1)
            if valid_length == 0:
                for row_index in row_indices.tolist():
                    pooled_rows[row_index] = temporal_tokens.new_zeros(
                        self.d_model
                    )
                continue

            valid_tokens = temporal_tokens.index_select(
                0,
                row_indices,
            )[:, -valid_length:, :]
            pooled = self._forward_blocks(valid_tokens)[:, -1, :]
            for output_index, row_index in enumerate(row_indices.tolist()):
                pooled_rows[row_index] = pooled[output_index]

        # Mamba is causal, so each final valid hidden state summarizes history.
        return torch.stack(pooled_rows, dim=0)
