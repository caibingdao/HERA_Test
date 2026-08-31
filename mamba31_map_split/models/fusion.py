import torch
import torch.nn as nn


class GraphMambaFusion(nn.Module):
    """Fuse one graph-state vector with one temporal-state vector."""

    def __init__(
        self,
        graph_dim: int,
        temporal_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        if min(graph_dim, temporal_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("Fusion dimensions must be positive")

        self.graph_dim = graph_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim + temporal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        graph_feature: torch.Tensor,
        temporal_feature: torch.Tensor,
    ) -> torch.Tensor:
        if graph_feature.dim() != 2 or temporal_feature.dim() != 2:
            raise ValueError("Fusion inputs must have shape [batch, dim]")
        if graph_feature.size(0) != temporal_feature.size(0):
            raise ValueError("Fusion inputs must have the same batch size")
        if graph_feature.size(-1) != self.graph_dim:
            raise ValueError(
                f"Expected graph feature dim {self.graph_dim}, "
                f"got {graph_feature.size(-1)}"
            )
        if temporal_feature.size(-1) != self.temporal_dim:
            raise ValueError(
                f"Expected temporal feature dim {self.temporal_dim}, "
                f"got {temporal_feature.size(-1)}"
            )

        fused = torch.cat((graph_feature, temporal_feature), dim=-1)
        return self.mlp(fused)
