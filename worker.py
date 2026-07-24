import torch

from env import Env
from agent import Agent
from utils import *
from model import PolicyNet
from replay_buffer import (
    EpisodeTrajectory,
    TrajectoryStep,
    build_history_sequence,
    detach_observation,
)

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(
        self,
        meta_agent_id,
        policy_net,
        global_step,
        device='cpu',
        save_image=False,
        rollout_history_cache=None,
        policy_version=0,
    ):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.rollout_history_cache = rollout_history_cache
        self.policy_version = policy_version

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

        self.use_mamba = getattr(policy_net, "use_mamba", False)
        self.trajectory = (
            EpisodeTrajectory(episode_id=global_step)
            if COLLECT_TRAJECTORY or self.use_mamba
            else None
        )
        if self.rollout_history_cache is not None and not self.use_mamba:
            raise ValueError(
                "Rollout history cache requires a Mamba policy"
            )
        self.robot_heading = 0.0

    def run_episode(self):
        if self.rollout_history_cache is not None:
            parameter = next(self.robot.policy_net.parameters())
            self.rollout_history_cache.start_episode(
                episode_id=self.global_step,
                policy_version=self.policy_version,
                device=parameter.device,
                dtype=parameter.dtype,
                use_mamba=self.use_mamba,
                sequence_length=self.robot.policy_net.mamba_seq_len,
                graph_embedding_dim=(
                    self.robot.policy_net
                    .history_encoder
                    .graph_embedding_dim
                ),
                token_dim=self.robot.policy_net.mamba_d_model,
            )

        try:
            return self._run_episode()
        finally:
            if (
                self.rollout_history_cache is not None
                and self.rollout_history_cache.active
            ):
                self.rollout_history_cache.end_episode(
                    self.global_step
                )

    def _run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()

        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):

            current_observation = observation
            if not self.use_mamba:
                self.save_observation(current_observation)

            coverage_before = self.env.get_exploration_rate()
            frontier_before = len(get_frontier_in_map(self.env.belief_info))
            robot_pose = torch.tensor(
                [
                    self.env.robot_location[0],
                    self.env.robot_location[1],
                    self.robot_heading,
                ],
                dtype=torch.float32,
            )

            rollout_context = None
            if self.rollout_history_cache is not None:
                (
                    next_location,
                    action_index,
                    rollout_context,
                ) = self.robot.select_next_waypoint_cached(
                    observation,
                    history_cache=self.rollout_history_cache,
                    episode_id=self.global_step,
                    policy_version=self.policy_version,
                    decision_step=i,
                )
            else:
                history_batch = None
                if self.use_mamba:
                    history_batch = build_history_sequence(
                        self.trajectory,
                        sequence_length=MAMBA_SEQ_LEN,
                    )
                    if history_batch is not None:
                        history_batch = history_batch.to(self.device)

                next_location, action_index = self.robot.select_next_waypoint(
                    observation,
                    history_batch=history_batch,
                )
            if not self.use_mamba:
                self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location, self.robot.location, node.data.neighbor_set)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            movement = next_location - self.env.robot_location
            next_heading = np.arctan2(movement[1], movement[0])
            reward = self.env.step(next_location)

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            if not self.use_mamba:
                self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            if not self.use_mamba:
                self.save_next_observations(observation)
            delta_coverage = self.env.explored_rate - coverage_before
            frontier_after = len(get_frontier_in_map(self.env.belief_info))
            delta_frontier = frontier_after - frontier_before

            if self.trajectory is not None:
                self.trajectory.append(
                    TrajectoryStep(
                        observation=detach_observation(
                            current_observation
                        ),
                        action=int(action_index.item()),
                        reward=float(reward),
                        next_observation=detach_observation(observation),
                        robot_pose=robot_pose,
                        delta_coverage=float(delta_coverage),
                        delta_frontier=int(delta_frontier),
                        done=bool(done),
                        step_index=i,
                    )
                )
            if self.rollout_history_cache is not None:
                (
                    graph_embedding,
                    action_feature,
                    history_token,
                ) = self.robot.policy_net.finalize_rollout_history_token(
                    rollout_context=rollout_context,
                    action_index=int(action_index.item()),
                    robot_pose=robot_pose,
                    reward=reward,
                    delta_coverage=delta_coverage,
                    delta_frontier=delta_frontier,
                )
                self.rollout_history_cache.append(
                    episode_id=self.global_step,
                    policy_version=self.policy_version,
                    transition_id=i,
                    graph_embedding=graph_embedding,
                    action_feature=action_feature,
                    token=history_token,
                )
            self.robot_heading = float(next_heading)

            if self.save_image:
                self.robot.plot_env()
                self.env.plot_env(i+1)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['policy_model'])
    worker = Worker(0, model, 77, save_image=True)
    worker.run_episode()
