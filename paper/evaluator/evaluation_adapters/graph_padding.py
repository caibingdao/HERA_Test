"""Evaluation-only graph padding; retain every node and every history feature."""
import copy
from dataclasses import replace
import torch.nn.functional as F


def observation(robot):
    method=robot.get_observation
    scope=method.__func__.__globals__
    original=int(scope['NODE_PADDING_SIZE'])
    needed=int(robot.node_coords.shape[0])+1  # original observation uses strict <
    capacity=max(original,int(getattr(robot,'_evaluation_padding_capacity',original)),((needed+31)//32)*32 if needed>original else original)
    robot._evaluation_padding_capacity=capacity
    scope['NODE_PADDING_SIZE']=capacity
    try:return method()
    finally:scope['NODE_PADDING_SIZE']=original


def pad_unbatched(graph,size):
    difference=size-graph[0].shape[0]
    if difference<0:raise ValueError('evaluation padding must never crop nodes')
    if not difference:return graph
    node,mask,edge,current,neighbors,neighbor_mask=graph
    return (F.pad(node,(0,0,0,difference)),F.pad(mask,(0,difference),value=True),
            F.pad(edge,(0,difference,0,difference),value=True),current,neighbors,neighbor_mask)


def history(builder,trajectory,sequence_length,decision_step=None):
    # The native history collator stacks node tensors. Align the last sequence
    # and its zero-reference with masked zero rows, without touching stored data.
    decision=len(trajectory) if decision_step is None else decision_step
    if not len(trajectory):return builder(trajectory,sequence_length,decision_step)
    selected={0,*range(max(0,decision-sequence_length),decision)}
    size=max(trajectory.steps[i].observation[0].shape[0] for i in selected)
    if all(trajectory.steps[i].observation[0].shape[0]==size for i in selected):
        return builder(trajectory,sequence_length,decision_step)
    aligned=copy.copy(trajectory);aligned.steps=list(trajectory.steps)
    for i in selected:aligned.steps[i]=replace(trajectory.steps[i],observation=pad_unbatched(trajectory.steps[i].observation,size))
    return builder(aligned,sequence_length,decision_step)
