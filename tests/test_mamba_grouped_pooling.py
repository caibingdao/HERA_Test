import copy
import unittest

import torch

from models.mamba_encoder import MambaTemporalEncoder


def legacy_grouped_pooling(encoder, temporal_tokens, padding_mask):
    padding_counts = padding_mask.sum(dim=1)
    valid_lengths = temporal_tokens.size(1) - padding_counts
    pooled_rows = [None] * temporal_tokens.size(0)
    for valid_length in torch.unique(valid_lengths).cpu().tolist():
        row_indices = torch.nonzero(
            valid_lengths == valid_length,
            as_tuple=False,
        ).squeeze(1)
        if valid_length == 0:
            for row_index in row_indices.cpu().tolist():
                pooled_rows[row_index] = temporal_tokens.new_zeros(
                    encoder.d_model
                )
            continue
        valid_tokens = temporal_tokens.index_select(
            0,
            row_indices,
        )[:, -valid_length:, :]
        pooled = encoder._forward_blocks(valid_tokens)[:, -1, :]
        for output_index, row_index in enumerate(
            row_indices.cpu().tolist()
        ):
            pooled_rows[row_index] = pooled[output_index]
    return torch.stack(pooled_rows, dim=0)


@unittest.skipUnless(torch.cuda.is_available(), "Mamba requires CUDA")
class MambaGroupedPoolingTest(unittest.TestCase):
    def compare_pattern(self, padding_mask):
        torch.manual_seed(20260725)
        device = torch.device("cuda")
        padding_mask = torch.tensor(
            padding_mask,
            dtype=torch.bool,
            device=device,
        )
        batch_size, sequence_length = padding_mask.shape
        baseline = MambaTemporalEncoder(d_model=8).to(device)
        optimized = copy.deepcopy(baseline)
        baseline_tokens = torch.randn(
            batch_size,
            sequence_length,
            8,
            device=device,
            requires_grad=True,
        )
        optimized_tokens = (
            baseline_tokens.detach().clone().requires_grad_(True)
        )

        baseline_output = legacy_grouped_pooling(
            baseline,
            baseline_tokens,
            padding_mask,
        )
        optimized_output = optimized(
            optimized_tokens,
            padding_mask,
        )
        torch.testing.assert_close(
            baseline_output,
            optimized_output,
            rtol=1e-6,
            atol=1e-7,
        )

        self.assertEqual(
            baseline_output.requires_grad,
            optimized_output.requires_grad,
        )
        if not baseline_output.requires_grad:
            return
        baseline_output.square().sum().backward()
        optimized_output.square().sum().backward()
        torch.testing.assert_close(
            baseline_tokens.grad,
            optimized_tokens.grad,
            rtol=1e-6,
            atol=1e-7,
        )
        for baseline_parameter, optimized_parameter in zip(
            baseline.parameters(),
            optimized.parameters(),
        ):
            torch.testing.assert_close(
                baseline_parameter.grad,
                optimized_parameter.grad,
                rtol=1e-6,
                atol=1e-7,
            )

    def test_all_same_length(self):
        self.compare_pattern(
            [
                [False, False, False, False],
                [False, False, False, False],
            ]
        )

    def test_mixed_valid_lengths(self):
        self.compare_pattern(
            [
                [True, True, True, False],
                [True, True, False, False],
                [False, False, False, False],
            ]
        )

    def test_all_padding(self):
        self.compare_pattern(
            [
                [True, True, True, True],
                [True, True, True, True],
            ]
        )


if __name__ == "__main__":
    unittest.main()
