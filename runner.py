import torch
import ray
from model import PolicyNet
from worker import Worker
from parameter import *


def validate_runner_resource_config(
    num_meta_agents=NUM_META_AGENT,
    runner_gpu_fraction=RUNNER_GPU_FRACTION,
):
    if isinstance(num_meta_agents, bool) or not isinstance(
        num_meta_agents,
        int,
    ):
        raise ValueError(
            "NUM_META_AGENT must be a positive integer, "
            f"got {num_meta_agents!r}"
        )
    if num_meta_agents <= 0:
        raise ValueError(
            f"NUM_META_AGENT must be > 0, got {num_meta_agents}"
        )
    if isinstance(runner_gpu_fraction, bool) or not isinstance(
        runner_gpu_fraction,
        (int, float),
    ):
        raise ValueError(
            "RUNNER_GPU_FRACTION must be a positive number, "
            f"got {runner_gpu_fraction!r}"
        )
    if runner_gpu_fraction <= 0:
        raise ValueError(
            "RUNNER_GPU_FRACTION must be > 0, "
            f"got {runner_gpu_fraction}"
        )

    total_runner_gpu = num_meta_agents * runner_gpu_fraction
    if total_runner_gpu > 1.0:
        raise ValueError(
            "Invalid Ray GPU allocation: NUM_META_AGENT "
            f"({num_meta_agents}) * RUNNER_GPU_FRACTION "
            f"({runner_gpu_fraction}) = {total_runner_gpu}, "
            "which must be <= 1.0"
        )
    return total_runner_gpu


validate_runner_resource_config()


class Runner(object):
    def __init__(self, meta_agent_id, use_mamba=None):
        self.meta_agent_id = meta_agent_id
        self.use_mamba = USE_MAMBA if use_mamba is None else use_mamba
        requires_cuda = USE_GPU or self.use_mamba
        if requires_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU rollout is required when USE_GPU or USE_MAMBA is enabled"
            )
        self.device = (
            torch.device('cuda') if requires_cuda else torch.device('cpu')
        )
        self.local_network = PolicyNet(
            NODE_INPUT_DIM,
            EMBEDDING_DIM,
            use_mamba=self.use_mamba,
            mamba_seq_len=MAMBA_SEQ_LEN,
            mamba_d_model=MAMBA_D_MODEL,
            mamba_d_state=MAMBA_D_STATE,
            mamba_d_conv=MAMBA_D_CONV,
            mamba_expand=MAMBA_EXPAND,
            fusion_hidden_dim=FUSION_HIDDEN_DIM,
            action_embedding_dim=ACTION_EMBEDDING_DIM,
        )
        self.local_network.to(self.device)

    def get_weights(self):
        return self.local_network.state_dict()

    def get_runtime_resources(self):
        return {
            "device_type": self.device.type,
            "ray_gpu_ids": ray.get_gpu_ids(),
        }

    def set_policy_net_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        # save_img = True
        worker = Worker(self.meta_agent_id, self.local_network, episode_number, device=self.device, save_image=save_img)
        worker.run_episode()

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        trajectory = worker.trajectory
        return job_results, perf_metrics, trajectory

    def job(self, weights_set, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_policy_net_weights(weights_set[0])

        job_results, metrics, trajectory = self.do_job(episode_number)

        info = {"id": self.meta_agent_id, "episode_number": episode_number}

        return job_results, metrics, trajectory, info


ACTIVE_RUNNER_GPU_FRACTION = (
    RUNNER_GPU_FRACTION if USE_GPU or USE_MAMBA else 0.0
)


@ray.remote(num_cpus=1, num_gpus=ACTIVE_RUNNER_GPU_FRACTION)
class RLRunner(Runner):
    def __init__(self, meta_agent_id, use_mamba=None):
        super().__init__(meta_agent_id, use_mamba=use_mamba)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
