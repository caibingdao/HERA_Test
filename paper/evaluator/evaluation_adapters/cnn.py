"""Matched CNN inference in the same seeded test Env as token21.

Loads only the CNN policy. No training workers, optimizers, or replay are created.
"""
from pathlib import Path
import importlib
import math
import sys
import time
import numpy as np
import torch
from env import Env
from experiment_utils import coverage_success, trajectory_path_length, sha256_file
from utils import get_frontier_in_map
from .classical import GEOMETRY_FIELDS
import parameter as shared_parameters

ROOT=Path(__file__).resolve().parents[3]/'ARiADNE_CNN'


def load_components():
    names={p.stem for p in ROOT.glob('*.py')}
    previous={name:sys.modules.pop(name) for name in names if name in sys.modules}
    old_path=sys.path[:]
    sys.path.insert(0,str(ROOT))
    try:
        training=importlib.import_module('matched_training')
        policy_class,_=training.build_networks()
        agent=importlib.import_module('agent')
        agent.USE_ANTI_LOOP_FILTER=False
        parameters=training.matched_parameters()
        return policy_class,agent.Agent,training.make_recovery,parameters
    finally:
        sys.path[:]=old_path
        for name in names:sys.modules.pop(name,None)
        sys.modules.update(previous)


class CNNAdapter:
    def __init__(self,checkpoint,deterministic,adapter_args):
        if not deterministic:raise ValueError('CNN test uses greedy argmax')
        if set(adapter_args)-{'device'}:raise ValueError('Unsupported CNN adapter arguments')
        self.path=Path(checkpoint).resolve()
        saved=torch.load(self.path,map_location='cpu',weights_only=False)
        if saved.get('model_type')!='CNN' or not saved.get('is_final'):
            raise ValueError('Use the completed matched CNN checkpoint')
        self.seed=int(saved['training_seed'])
        self.episodes=int(saved['completed_episodes'])
        self.recovery_enabled=bool(saved['use_local_revisit_recovery'])
        if not self.recovery_enabled:raise ValueError('Matched CNN checkpoint expected recovery on')
        self.device=torch.device(adapter_args.get('device','cuda'))
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CNN test requires CUDA')
        policy_class,self.Agent,self.make_recovery,p=load_components()
        for field in GEOMETRY_FIELDS:
            if getattr(p,field)!=getattr(shared_parameters,field):raise ValueError(f'CNN geometry mismatch: {field}')
        for key,value in saved['cnn_config'].items():
            if getattr(p,key)!=value:raise ValueError(f'CNN architecture mismatch: {key}')
        self.policy=policy_class().to(self.device).eval()
        self.policy.load_state_dict(saved['policy_model'],strict=True)
        self.parameters=p
        controller=self.make_recovery()
        if not controller.enabled:raise ValueError('CNN recovery unexpectedly disabled')
        self.code_hashes={x.name:sha256_file(x) for x in ROOT.glob('*.py')}
        self.diagnostics=[]

    def metadata(self):
        return dict(model_name='ARiADNE CNN matched recovery on',model_seed=self.seed,
                    model_type='CNN',use_mamba=False,use_cnn=True,trained=True,
                    completed_training_episodes=self.episodes,device=str(self.device),
                    deterministic=True,action_selection='greedy_argmax',
                    recovery_enabled=True,recovery_controller='local_revisit_recovery_v4',
                    anti_loop_filter=False,checkpoint=str(self.path),source_sha256=self.code_hashes,
                    movement_validation="maintained graph adjacency and shared Env.step, matching CNN training and Mamba evaluation",
                    evaluation_adapter_sha256=sha256_file(Path(__file__)),
                    cnn_config={k:v for k,v in vars(self.parameters).items() if k.startswith('CNN_')},
                    environment='shared Mamba split-aware Env',episode_diagnostics=self.diagnostics)

    def evaluate_episode(self,*,map_path,map_id,eval_seed,start_seed,episode_id,config):
        started=time.perf_counter();target=float(config['target_coverage'])
        env=Env(episode_id,plot=False,map_path=map_path,start_seed=start_seed)
        robot=self.Agent(self.policy,self.device,plot=False)
        robot.update_planning_state(env.belief_info,env.robot_location)
        recovery=self.make_recovery()
        initial=float(env.get_exploration_rate());curve=[dict(step=0,path_length=0.,coverage=initial)]
        steps=0;timings=[];reason=None
        environment_success=bool(robot.utility.sum()==0)
        if coverage_success(initial,target):reason='target_reached'
        elif environment_success:reason='environment_complete'
        while reason is None and steps<int(config['max_steps']):
            observation=robot.get_observation_cnn()
            before=float(env.get_exploration_rate())
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            decision_started=time.perf_counter()
            decision=recovery.begin_step(robot.location,robot.node_manager,
                frontier_points=get_frontier_in_map(env.belief_info),
                exploration_complete=bool(robot.utility.sum()==0))
            if decision.is_recovery_action:
                next_location=np.asarray(decision.next_waypoint)
                indices=observation[4][0,:,0].cpu().numpy()
                valid=observation[3][0].cpu().numpy()>0
                matches=[slot for slot,node in enumerate(indices) if valid[slot] and
                         np.allclose(robot.node_coords[node],next_location,rtol=0,atol=1e-6)]
                if not matches:raise RuntimeError('Recovery action absent from CNN candidate slots')
            else:
                with torch.inference_mode():
                    logp=self.policy(*observation[:4])
                    masked=logp.masked_fill(observation[3]<=0,-torch.inf)
                    if not torch.isfinite(masked).any():
                        reason='no_valid_action';break
                    slot=int(masked.argmax(dim=1).item())
                index=int(observation[4][0,slot,0].item())
                if not 0<=index<len(robot.node_coords):raise RuntimeError('CNN selected invalid node index')
                next_location=np.asarray(robot.node_coords[index])
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            timings.append(time.perf_counter()-decision_started)
            if next_location.shape!=(2,) or not np.isfinite(next_location).all():raise ValueError('Malformed CNN waypoint')
            if np.allclose(env.robot_location,next_location,rtol=0,atol=1e-9):reason='no_valid_action';break
            node=robot.node_manager.check_node_exist_in_dict(tuple(robot.location)).data
            neighbors={tuple(np.round(p,1)) for p in node.neighbor_set}
            if tuple(np.round(next_location,1)) not in neighbors:raise RuntimeError('Nonadjacent CNN waypoint')
            env.step(next_location);steps+=1
            robot.update_planning_state(env.belief_info,env.robot_location)
            coverage=float(env.get_exploration_rate())
            recovery.record_coverage_gain(coverage-before)
            environment_success=bool(robot.utility.sum()==0)
            curve.append(dict(step=steps,path_length=float(env.travel_dist),coverage=coverage))
            if coverage_success(coverage,target):reason='target_reached'
            elif environment_success:reason='environment_complete'
            elif config['max_path_length'] is not None and env.travel_dist>=config['max_path_length']:reason='max_path_length'
        reason=reason or 'max_steps';recovery.end_episode(env.robot_location)
        distance=float(env.travel_dist);trajectory_distance=trajectory_path_length(env.robot_trajectory)
        if len(env.robot_trajectory)!=steps+1 or not math.isclose(distance,trajectory_distance,abs_tol=1e-8):
            raise RuntimeError('CNN trajectory length inconsistency')
        coverage=float(env.get_exploration_rate());success=coverage_success(coverage,target)
        self.diagnostics.append(dict(map_id=map_id,recovery_steps=recovery.recovery_steps,
                                     recovery_triggers=recovery.recovery_trigger_count))
        return dict(episode=dict(start_x=0.,start_y=0.,start_cell_x=int(env.start_cell[0]),
            start_cell_y=int(env.start_cell[1]),start_resample_count=int(env.start_resample_count),
            success=success,coverage_success=success,environment_success=environment_success,
            termination_reason=reason,path_length=distance,steps=steps,initial_coverage=initial,
            final_coverage=coverage,target_coverage=target,
            path_length_to_target=distance if success else math.nan,steps_to_target=steps if success else math.nan,
            elapsed_time_seconds=time.perf_counter()-started,
            mean_inference_time_ms=float(np.mean(timings)*1000) if timings else math.nan,
            max_inference_time_ms=max(timings)*1000 if timings else math.nan,
            trajectory_path_length=trajectory_distance,path_length_validation_error=abs(distance-trajectory_distance),
            recovery_enabled=True,recovery_steps=recovery.recovery_steps,recovery_trigger_count=recovery.recovery_trigger_count,error=''),
            coverage_curve=curve)


def create_adapter(checkpoint,deterministic,adapter_args):
    return CNNAdapter(checkpoint,deterministic,adapter_args)
