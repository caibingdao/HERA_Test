import unittest

import torch

from replay_buffer import TrajectoryReplayBuffer
from tests.test_temporal_pipeline import make_episode


def assert_history_equal(test_case, baseline, union):
    for baseline_tensor, union_tensor in zip(
        baseline.observations,
        union.observations,
    ):
        torch.testing.assert_close(baseline_tensor, union_tensor)
    for field in (
        "actions",
        "rewards",
        "robot_pose",
        "delta_coverage",
        "delta_frontier",
        "padding_mask",
        "episode_ids",
        "decision_steps",
    ):
        torch.testing.assert_close(
            getattr(baseline, field),
            getattr(union, field),
        )


class AlignedUnionHistoryTest(unittest.TestCase):
    def build_replay(self, enabled):
        replay = TrajectoryReplayBuffer(
            capacity_steps=20,
            use_union_history_collate=enabled,
        )
        replay.add_episode(make_episode(episode_id=10, offset=100))
        replay.add_episode(make_episode(episode_id=20, offset=200))
        return replay

    def test_union_matches_legacy_for_all_early_step_cases(self):
        baseline_replay = self.build_replay(enabled=False)
        union_replay = self.build_replay(enabled=True)
        baseline = baseline_replay.sample_aligned_transitions(
            batch_size=10,
            sequence_length=3,
            seed=9,
        )
        union = union_replay.sample_aligned_transitions(
            batch_size=10,
            sequence_length=3,
            seed=9,
        )

        self.assertIsNotNone(union.union_history)
        assert_history_equal(
            self,
            baseline.current_history,
            union.current_history,
        )
        assert_history_equal(
            self,
            baseline.next_history,
            union.next_history,
        )
        for baseline_tensor, union_tensor in zip(
            baseline.observations,
            union.observations,
        ):
            torch.testing.assert_close(baseline_tensor, union_tensor)
        for field in (
            "actions",
            "rewards",
            "dones",
            "episode_ids",
            "step_indices",
        ):
            torch.testing.assert_close(
                getattr(baseline, field),
                getattr(union, field),
            )

    def test_current_and_next_are_views_of_one_union_transfer(self):
        batch = self.build_replay(
            enabled=True,
        ).sample_aligned_transitions(
            batch_size=4,
            sequence_length=3,
            seed=5,
        )
        moved = batch.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.assertIsNotNone(moved.union_history)
        self.assertEqual(
            moved.current_history.actions.untyped_storage().data_ptr(),
            moved.next_history.actions.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            moved.current_history.padding_mask.untyped_storage().data_ptr(),
            moved.next_history.padding_mask.untyped_storage().data_ptr(),
        )
        for current, next_value in zip(
            moved.current_history.observations,
            moved.next_history.observations,
        ):
            self.assertEqual(
                current.untyped_storage().data_ptr(),
                next_value.untyped_storage().data_ptr(),
            )


if __name__ == "__main__":
    unittest.main()
