import math
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from training_profiler import profile_section


class HistoryStateEncoder(nn.Module):
    """Build a fixed-width temporal token from one exploration transition."""

    def __init__(
        self,
        graph_embedding_dim: int,
        action_embedding_dim: int = 16,
        token_dim: int = 128,
        pose_scale: float = 80.0,
        reward_scale: float = 20.0,
        delta_frontier_scale: float = 360.0,
        graph_batch_size: int = 16,
        skip_padded_history_graphs: bool = False,
        trim_history_graph_nodes: bool = False,
        use_history_checkpoint: bool = True,
    ):
        super().__init__()
        if graph_embedding_dim <= 0:
            raise ValueError("graph_embedding_dim must be positive")
        if action_embedding_dim <= 0 or token_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        if (
            pose_scale <= 0
            or reward_scale <= 0
            or delta_frontier_scale <= 0
        ):
            raise ValueError("normalization scales must be positive")
        if graph_batch_size <= 0:
            raise ValueError("graph_batch_size must be positive")

        self.graph_embedding_dim = graph_embedding_dim
        self.token_dim = token_dim
        self.pose_scale = pose_scale
        self.reward_scale = reward_scale
        self.delta_frontier_log_scale = math.log1p(delta_frontier_scale)
        self.graph_batch_size = graph_batch_size
        self.skip_padded_history_graphs = bool(
            skip_padded_history_graphs
        )
        self.trim_history_graph_nodes = bool(trim_history_graph_nodes)
        self.use_history_checkpoint = bool(use_history_checkpoint)

        self.action_projection = nn.Linear(
            graph_embedding_dim,
            action_embedding_dim,
        )

        # pose(3), reward(1), coverage delta(1), frontier-count delta(1)
        input_dim = graph_embedding_dim + action_embedding_dim + 6
        self.input_norm = nn.LayerNorm(input_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.output_norm = nn.LayerNorm(token_dim)

    @staticmethod
    def pool_graph(
        enhanced_node_feature: torch.Tensor,
        node_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked mean pooling over valid graph nodes."""
        if enhanced_node_feature.dim() != 3:
            raise ValueError(
                "enhanced_node_feature must have shape [batch, nodes, dim]"
            )

        if node_padding_mask.dim() == 3:
            if node_padding_mask.size(1) != 1:
                raise ValueError(
                    "node_padding_mask must have shape [batch, 1, nodes]"
                )
            node_padding_mask = node_padding_mask.squeeze(1)
        elif node_padding_mask.dim() != 2:
            raise ValueError(
                "node_padding_mask must have shape [batch, nodes] or "
                "[batch, 1, nodes]"
            )

        if enhanced_node_feature.shape[:2] != node_padding_mask.shape:
            raise ValueError(
                "Node feature and padding mask dimensions do not match"
            )

        valid_nodes = (~node_padding_mask.bool()).unsqueeze(-1)
        valid_nodes = valid_nodes.to(enhanced_node_feature.dtype)
        valid_count = valid_nodes.sum(dim=1).clamp_min(1.0)
        return (enhanced_node_feature * valid_nodes).sum(dim=1) / valid_count

    @staticmethod
    def _encode_graph_chunk(
        graph_model: nn.Module,
        node_inputs: torch.Tensor,
        node_padding_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        current_edge: torch.Tensor,
        actions: torch.Tensor,
    ):
        enhanced_node_feature = graph_model.encode_graph(
            node_inputs,
            node_padding_mask,
            edge_mask,
        )
        graph_embedding = HistoryStateEncoder.pool_graph(
            enhanced_node_feature,
            node_padding_mask,
        )
        selected_node_index = torch.gather(
            current_edge,
            dim=1,
            index=actions,
        )
        selected_action_feature = torch.gather(
            enhanced_node_feature,
            dim=1,
            index=selected_node_index.repeat(
                1,
                1,
                enhanced_node_feature.size(-1),
            ),
        ).squeeze(1)
        return graph_embedding, selected_action_feature

    def graph_features_from_observations(
        self,
        graph_model: nn.Module,
        observations: Sequence[torch.Tensor],
        actions: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        """Encode global graph states and selected neighboring action nodes."""
        if len(observations) != 6:
            raise ValueError(f"Expected 6 observation tensors, got {len(observations)}")

        node_inputs, node_padding_mask, edge_mask = observations[:3]
        current_edge = observations[4]
        if node_inputs.dim() != 4:
            raise ValueError(
                "Sequence node inputs must have shape "
                "[batch, sequence, nodes, features]"
            )

        batch_size, sequence_length, node_count, node_dim = node_inputs.shape
        if (
            padding_mask is not None
            and padding_mask.shape != (batch_size, sequence_length)
        ):
            raise ValueError(
                "padding_mask must have shape [batch, sequence]"
            )
        flat_node_inputs = node_inputs.reshape(
            batch_size * sequence_length, node_count, node_dim
        )
        flat_node_padding_mask = node_padding_mask.reshape(
            batch_size * sequence_length, 1, node_count
        )
        flat_edge_mask = edge_mask.reshape(
            batch_size * sequence_length, node_count, node_count
        )

        flat_current_edge = current_edge.reshape(
            batch_size * sequence_length,
            current_edge.size(2),
            1,
        ).long()
        flat_actions = actions.reshape(
            batch_size * sequence_length,
            1,
            1,
        ).long()
        if torch.any(flat_actions < 0) or torch.any(
            flat_actions >= flat_current_edge.size(1)
        ):
            raise ValueError("Action index is outside the current neighbor list")

        flat_batch_size = flat_node_inputs.size(0)
        valid_indices = None
        if self.skip_padded_history_graphs and padding_mask is not None:
            valid_indices = torch.nonzero(
                ~padding_mask.bool().reshape(-1),
                as_tuple=False,
            ).squeeze(1)
            if valid_indices.numel() == 0:
                empty_shape = (
                    batch_size,
                    sequence_length,
                    self.graph_embedding_dim,
                )
                return (
                    flat_node_inputs.new_zeros(empty_shape),
                    flat_node_inputs.new_zeros(empty_shape),
                )
            flat_node_inputs = flat_node_inputs.index_select(
                0,
                valid_indices,
            )
            flat_node_padding_mask = flat_node_padding_mask.index_select(
                0,
                valid_indices,
            )
            flat_edge_mask = flat_edge_mask.index_select(0, valid_indices)
            flat_current_edge = flat_current_edge.index_select(
                0,
                valid_indices,
            )
            flat_actions = flat_actions.index_select(0, valid_indices)

        graph_chunks = []
        action_chunks = []
        encoded_batch_size = flat_node_inputs.size(0)
        for start in range(0, encoded_batch_size, self.graph_batch_size):
            end = min(start + self.graph_batch_size, encoded_batch_size)

            def encode_chunk(
                chunk_node_inputs,
                chunk_node_padding_mask,
                chunk_edge_mask,
                chunk_current_edge,
                chunk_actions,
            ):
                return self._encode_graph_chunk(
                    graph_model,
                    chunk_node_inputs,
                    chunk_node_padding_mask,
                    chunk_edge_mask,
                    chunk_current_edge,
                    chunk_actions,
                )

            chunk_inputs = (
                flat_node_inputs[start:end],
                flat_node_padding_mask[start:end],
                flat_edge_mask[start:end],
                flat_current_edge[start:end],
                flat_actions[start:end],
            )
            if self.trim_history_graph_nodes:
                (
                    chunk_node_inputs,
                    chunk_node_padding_mask,
                    chunk_edge_mask,
                    chunk_current_edge,
                    chunk_actions,
                ) = chunk_inputs
                valid_node_count = (
                    ~chunk_node_padding_mask.squeeze(1).bool()
                ).sum(dim=1)
                chunk_max_nodes = max(
                    int(valid_node_count.max().item()),
                    1,
                )
                selected_node_index = torch.gather(
                    chunk_current_edge,
                    dim=1,
                    index=chunk_actions,
                )
                if torch.any(selected_node_index >= chunk_max_nodes):
                    raise ValueError(
                        "Selected action node lies outside the dynamically "
                        "trimmed history graph"
                    )
                chunk_inputs = (
                    chunk_node_inputs[:, :chunk_max_nodes],
                    chunk_node_padding_mask[:, :, :chunk_max_nodes],
                    chunk_edge_mask[
                        :,
                        :chunk_max_nodes,
                        :chunk_max_nodes,
                    ],
                    chunk_current_edge,
                    chunk_actions,
                )
            if (
                self.use_history_checkpoint
                and torch.is_grad_enabled()
                and graph_model.training
            ):
                graph_chunk, action_chunk = checkpoint(
                    encode_chunk,
                    *chunk_inputs,
                    use_reentrant=False,
                )
            else:
                graph_chunk, action_chunk = encode_chunk(*chunk_inputs)
            graph_chunks.append(graph_chunk)
            action_chunks.append(action_chunk)

        graph_embedding = torch.cat(graph_chunks, dim=0)
        selected_action_feature = torch.cat(action_chunks, dim=0)

        if valid_indices is not None:
            graph_embedding = graph_embedding.new_zeros(
                flat_batch_size,
                graph_embedding.size(-1),
            ).index_copy(0, valid_indices, graph_embedding)
            selected_action_feature = selected_action_feature.new_zeros(
                flat_batch_size,
                selected_action_feature.size(-1),
            ).index_copy(
                0,
                valid_indices,
                selected_action_feature,
            )

        graph_embedding = graph_embedding.reshape(
            batch_size,
            sequence_length,
            -1,
        )
        selected_action_feature = selected_action_feature.reshape(
            batch_size,
            sequence_length,
            -1,
        )
        return graph_embedding, selected_action_feature

    def forward(
        self,
        graph_embedding: torch.Tensor,
        robot_pose: torch.Tensor,
        action_feature: torch.Tensor,
        reward: torch.Tensor,
        delta_coverage: torch.Tensor,
        delta_frontier: torch.Tensor,
    ) -> torch.Tensor:
        if graph_embedding.size(-1) != self.graph_embedding_dim:
            raise ValueError(
                f"Expected graph embedding dim {self.graph_embedding_dim}, "
                f"got {graph_embedding.size(-1)}"
            )
        if robot_pose.shape != graph_embedding.shape[:-1] + (3,):
            raise ValueError(
                "robot_pose leading dimensions must match graph_embedding"
            )
        if action_feature.shape != graph_embedding.shape:
            raise ValueError(
                "action_feature must match the graph_embedding shape"
            )

        reward = self._ensure_scalar_dimension(reward, graph_embedding)
        delta_coverage = self._ensure_scalar_dimension(
            delta_coverage, graph_embedding
        )
        delta_frontier = self._ensure_scalar_dimension(
            delta_frontier, graph_embedding
        )

        action_embedding = self.action_projection(action_feature)

        xy = robot_pose[..., :2] / self.pose_scale
        theta = robot_pose[..., 2:3]
        theta = torch.atan2(torch.sin(theta), torch.cos(theta)) / math.pi
        normalized_pose = torch.cat((xy, theta), dim=-1)

        normalized_reward = reward.float() / self.reward_scale
        normalized_coverage = delta_coverage.float()
        delta_frontier = delta_frontier.float()
        normalized_delta_frontier = (
            torch.sign(delta_frontier)
            * torch.log1p(torch.abs(delta_frontier))
            / self.delta_frontier_log_scale
        )

        state = torch.cat(
            (
                graph_embedding,
                normalized_pose,
                action_embedding,
                normalized_reward,
                normalized_coverage,
                normalized_delta_frontier,
            ),
            dim=-1,
        )
        token = self.token_projection(self.input_norm(state))
        return self.output_norm(token)

    def encode_sequence(self, graph_model: nn.Module, sequence_batch):
        with profile_section("history_graph_encode_ms"):
            graph_embedding, action_feature = (
                self.graph_features_from_observations(
                    graph_model,
                    sequence_batch.observations,
                    sequence_batch.actions,
                    padding_mask=getattr(
                        sequence_batch,
                        "padding_mask",
                        None,
                    ),
                )
            )
        with profile_section("history_token_projection_ms"):
            temporal_tokens = self(
                graph_embedding=graph_embedding,
                robot_pose=sequence_batch.robot_pose,
                action_feature=action_feature,
                reward=sequence_batch.rewards,
                delta_coverage=sequence_batch.delta_coverage,
                delta_frontier=sequence_batch.delta_frontier,
            )
        padding_mask = getattr(sequence_batch, "padding_mask", None)
        if padding_mask is not None:
            if padding_mask.shape != temporal_tokens.shape[:2]:
                raise ValueError(
                    "Temporal padding mask must have shape [batch, sequence]"
                )
            temporal_tokens = temporal_tokens.masked_fill(
                padding_mask.bool().unsqueeze(-1),
                0,
            )
        return temporal_tokens

    @staticmethod
    def _ensure_scalar_dimension(
        value: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        leading_shape = graph_embedding.shape[:-1]
        if value.shape == leading_shape:
            value = value.unsqueeze(-1)
        if value.shape != leading_shape + (1,):
            raise ValueError(
                "Scalar feature leading dimensions must match graph_embedding"
            )
        return value
