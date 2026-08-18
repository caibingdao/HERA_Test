import unittest

import torch

from models.history_encoder import HistoryStateEncoder
from models.mamba_encoder import (
    MAMBA_SSM_VERSION,
    MambaTemporalEncoder,
    OfficialMamba,
    _MAMBA_IMPORT_ERROR,
)
from parameter import (
    ACTION_EMBEDDING_DIM,
    EMBEDDING_DIM,
    TEMPORAL_SEQUENCE_LENGTH,
    TEMPORAL_TOKEN_DIM,
)
from replay_buffer import TrajectoryReplayBuffer
from runner import Runner


MAMBA_RUNTIME_READY = OfficialMamba is not None and torch.cuda.is_available()
MAMBA_SKIP_REASON = (
    "Official mamba-ssm and a CUDA GPU are required. "
    f"Import error: {_MAMBA_IMPORT_ERROR!r}"
)


@unittest.skipUnless(MAMBA_RUNTIME_READY, MAMBA_SKIP_REASON)
class MambaTemporalEncoderTest(unittest.TestCase):
    device = torch.device("cuda")

    def assert_backward_is_finite(self, inputs, outputs, model):
        loss = outputs.square().mean()
        loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

        parameter_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(parameter_gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
        )

    def test_random_temporal_tokens_forward_and_backward(self):
        batch_size = 2
        sequence_length = 8
        temporal_tokens = torch.randn(
            batch_size,
            sequence_length,
            TEMPORAL_TOKEN_DIM,
            device=self.device,
            requires_grad=True,
        )
        encoder = MambaTemporalEncoder(
            d_model=TEMPORAL_TOKEN_DIM,
        ).to(self.device)

        temporal_feature = encoder(temporal_tokens)

        self.assertEqual(
            temporal_feature.shape,
            (batch_size, TEMPORAL_TOKEN_DIM),
        )
        self.assertTrue(torch.isfinite(temporal_feature).all())
        self.assert_backward_is_finite(
            temporal_tokens,
            temporal_feature,
            encoder,
        )
        print(
            "random_mamba_result",
            {
                "mamba_ssm_version": MAMBA_SSM_VERSION,
                "input_shape": tuple(temporal_tokens.shape),
                "output_shape": tuple(temporal_feature.shape),
                "finite": True,
                "backward": True,
            },
        )

    def test_left_padding_mask_uses_only_valid_history(self):
        temporal_tokens = torch.randn(
            3,
            TEMPORAL_SEQUENCE_LENGTH,
            TEMPORAL_TOKEN_DIM,
            device=self.device,
            requires_grad=True,
        )
        padding_mask = torch.tensor(
            [
                [True] * (TEMPORAL_SEQUENCE_LENGTH - 3)
                + [False] * 3,
                [False] * TEMPORAL_SEQUENCE_LENGTH,
                [True] * TEMPORAL_SEQUENCE_LENGTH,
            ],
            device=self.device,
        )
        encoder = MambaTemporalEncoder(
            d_model=TEMPORAL_TOKEN_DIM,
        ).to(self.device)

        masked_output = encoder(
            temporal_tokens,
            padding_mask=padding_mask,
        )
        first_reference = encoder(temporal_tokens[0:1, -3:, :])
        second_reference = encoder(temporal_tokens[1:2])

        torch.testing.assert_close(
            masked_output[0:1],
            first_reference,
        )
        torch.testing.assert_close(
            masked_output[1:2],
            second_reference,
        )
        torch.testing.assert_close(
            masked_output[2],
            torch.zeros_like(masked_output[2]),
        )

        masked_output.square().mean().backward()
        self.assertTrue(torch.isfinite(temporal_tokens.grad).all())
        self.assertTrue(
            torch.all(
                temporal_tokens.grad[
                    0,
                    : TEMPORAL_SEQUENCE_LENGTH - 3,
                ]
                == 0
            )
        )
        self.assertTrue(torch.all(temporal_tokens.grad[2] == 0))

    def test_real_episode_temporal_tokens_forward_and_backward(self):
        runner = Runner(meta_agent_id=0, use_mamba=False)
        _, _, trajectory = runner.do_job(episode_number=77)

        replay = TrajectoryReplayBuffer(capacity_steps=10000)
        replay.add_episode(trajectory)
        sequence_batch = replay.sample_sequences(
            batch_size=1,
            sequence_length=TEMPORAL_SEQUENCE_LENGTH,
            seed=11,
        ).to(next(runner.local_network.parameters()).device)

        history_encoder = HistoryStateEncoder(
            graph_embedding_dim=EMBEDDING_DIM,
            action_embedding_dim=ACTION_EMBEDDING_DIM,
            token_dim=TEMPORAL_TOKEN_DIM,
        ).to(next(runner.local_network.parameters()).device)
        with torch.no_grad():
            temporal_tokens = history_encoder.encode_sequence(
                runner.local_network,
                sequence_batch,
            )
        temporal_tokens = temporal_tokens.to(self.device).requires_grad_(True)

        mamba_encoder = MambaTemporalEncoder(
            d_model=TEMPORAL_TOKEN_DIM,
        ).to(self.device)
        temporal_feature = mamba_encoder(temporal_tokens)

        self.assertEqual(
            temporal_feature.shape,
            (1, TEMPORAL_TOKEN_DIM),
        )
        self.assertTrue(torch.isfinite(temporal_feature).all())
        self.assert_backward_is_finite(
            temporal_tokens,
            temporal_feature,
            mamba_encoder,
        )
        print(
            "real_episode_mamba_result",
            {
                "mamba_ssm_version": MAMBA_SSM_VERSION,
                "episode_steps": len(trajectory),
                "input_shape": tuple(temporal_tokens.shape),
                "output_shape": tuple(temporal_feature.shape),
                "finite": True,
                "backward": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
