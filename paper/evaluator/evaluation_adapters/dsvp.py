"""Official-source DSVP planar port in the shared held-out exploration protocol."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from env import Env
from utils import get_frontier_in_map,check_collision
from experiment_utils import sha256_file,coverage_success,trajectory_path_length

WORKSPACE=Path(__file__).resolve().parents[3]

class DSVPAdapter:
    def __init__(self,checkpoint,deterministic,adapter_args):
        if not deterministic or adapter_args:raise ValueError('DSVP requires deterministic harness, configuration in specification')
        self.spec=json.loads(Path(checkpoint).read_text())
        if self.spec['planner']!='dsvp_planar':raise ValueError('wrong planner specification')
        root=WORKSPACE/'ARiADNE_DSVP'
        for filename,digest in self.spec['source_sha256'].items():
            if sha256_file(root/filename)!=digest:raise ValueError(f'source mismatch: {filename}')
        location=root/'planar'
        old=sys.path[:];sys.path.insert(0,str(location))
        try:
            module_spec=importlib.util.spec_from_file_location('dsvp_planar',location/'planner.py')
            module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        finally:sys.path[:]=old
        self.Planner=module.Planner;self.shortest=module.shortest;self.route=module.route
        self.diagnostics=[]

    def metadata(self):
        return dict(model_name='DSVP planar official-score port',model_seed=0,trained=False,
            planner='dsvp_planar',source_commit=self.spec['source_commit'],source_sha256=self.spec['source_sha256'],
            official_source=self.spec['official_source'],adaptation=self.spec['adaptation'],
            native_parameters=self.spec['configuration'],recovery_enabled=False,
            action_selection='seeded_dynamic_rrt_official_gain_and_relocation',
            start_position_mode='seeded_map_marker_selection',start_seed_applied=True,
            episode_diagnostics=self.diagnostics)

    def _execute_route(self,planner,native_route):
        # Follow the actual native route, not a projection onto a coarse graph.
        root=native_route[0];distance=0.;selected=None
        for a,b in zip(native_route,native_route[1:]):
            segment=float(np.linalg.norm(b-a))
            if segment<=1e-9:continue
            remaining=8.-distance
            point=a+(b-a)*min(1.,max(0.,remaining)/segment)
            if planner.grid.safe(root,point):selected=point
            else:break
            distance+=segment
            if distance>=8.:break
        if selected is None or np.linalg.norm(selected-root)<1e-6:return None
        planner.counters['native_path_execution_steps']+=1
        return np.asarray(selected)

    def evaluate_episode(self,*,map_path,map_id,eval_seed,start_seed,episode_id,config):
        started=time.perf_counter();environment=Env(episode_id,plot=False,map_path=map_path,start_seed=start_seed)
        planner=self.Planner()
        initial=float(environment.get_exploration_rate());target=float(config['target_coverage'])
        curve=[dict(step=0,path_length=0.,coverage=initial)];timings=[];steps=0;reason=None;stages={}
        if coverage_success(initial,target):reason='target_reached'
        while reason is None and steps<int(config['max_steps']):
            begin=time.perf_counter()
            frontiers=get_frontier_in_map(environment.belief_info)
            decision=planner.plan(environment.robot_location,frontiers,environment.belief_info)
            if decision is None:
                reason='no_valid_action';break
            goal,native_route=decision
            location=self._execute_route(planner,native_route)
            timings.append(time.perf_counter()-begin)
            stage=planner.last['stage'];stages[stage]=stages.get(stage,0)+1
            if location is None:
                reason='native_path_execution_failed';break
            if check_collision(environment.robot_location,location,environment.belief_info):
                raise RuntimeError('DSVP returned a colliding native path action')
            environment.step(location);steps+=1
            coverage=float(environment.get_exploration_rate())
            curve.append(dict(step=steps,path_length=float(environment.travel_dist),coverage=coverage))
            if coverage_success(coverage,target):reason='target_reached'
            elif config['max_path_length'] is not None and environment.travel_dist>=config['max_path_length']:reason='max_path_length'
        reason=reason or 'max_steps';coverage=float(environment.get_exploration_rate());success=coverage_success(coverage,target)
        distance=float(environment.travel_dist);trajectory=trajectory_path_length(environment.robot_trajectory)
        if not math.isclose(distance,trajectory,abs_tol=1e-8) or len(environment.robot_trajectory)!=steps+1:raise RuntimeError('trajectory accounting mismatch')
        self.diagnostics.append(dict(map_id=map_id,termination_reason=reason,counters=dict(planner.counters),stages=stages,last=planner.last))
        return {'episode':dict(start_x=0.,start_y=0.,start_cell_x=int(environment.start_cell[0]),start_cell_y=int(environment.start_cell[1]),start_resample_count=int(environment.start_resample_count),success=success,coverage_success=success,environment_success=False,termination_reason=reason,path_length=distance,steps=steps,initial_coverage=initial,final_coverage=coverage,target_coverage=target,path_length_to_target=distance if success else math.nan,steps_to_target=steps if success else math.nan,elapsed_time_seconds=time.perf_counter()-started,mean_inference_time_ms=float(np.mean(timings)*1000) if timings else math.nan,max_inference_time_ms=max(timings)*1000 if timings else math.nan,trajectory_path_length=trajectory,path_length_validation_error=abs(distance-trajectory),recovery_enabled=False,error=''), 'coverage_curve':curve}

def create_adapter(checkpoint,deterministic,adapter_args):return DSVPAdapter(checkpoint,deterministic,adapter_args)
