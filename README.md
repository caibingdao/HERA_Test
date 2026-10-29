# ARiADNE
Public code and model of <a href="https://arxiv.org/pdf/2301.11575.pdf">ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration</a>, which is accepted for the oral presentation at ICRA 2023.

**Note**: This is a new implementation of ARiADNE. 
You can find our original implementation in the [main branch](https://github.com/marmotlab/ARiADNE/tree/main).
We reimplement ARiADNE to optimize the computing time, RAM/VRAM usage, and compatibility with ROS. 
The trained model can be directly tested in our [ARiADNE ROS planner](https://github.com/marmotlab/ARiADNE-ROS-Planner).


## Run

#### Dependencies
We recommend to use conda for package management. 
Our planner is coded in Python and based on Pytorch. 
Other than Pytorch, please install following packages by:
```
pip install scikit-image matplotlib ray tensorboard
```
We tested our planner in various version of these packages so you can just install the latest one.

#### Training
Download this repo and go into the folder:
```
git clone https://github.com/marmotlab/ARiADNE.git
cd ARiADNE
```
Launch your conda environment if any and run:

```python driver.py```

The default training code requires around 8GB VRAM and 20G RAM. 
You can modify the hyperparameters in `parameter.py`.


## Files
* `parameters.py` Training parameters.
* `driver.py` Driver of training program, maintain & update the global network.
* `runner.py` Wrapper of the workers.
* `worker.py` Interact with environment and collect episode experience.
* `model.py` Define attention-based network.
* `env.py` Autonomous exploration environment.
* `node_manager.py` Manage and update the informative graph.
* `quads` Quad tree for node indexing provided by [Daniel Lindsley](https://github.com/toastdriven).
* `sensor.py` Simulate the sensor model of Lidar.
* `/maps` Maps of training environments provided by <a href="https://github.com/RobustFieldAutonomyLab/DRL_robot_exploration">Chen et al.</a>.

### Demo of ARiADNE

<div>
   <img src="gifs/demo_1.gif" width="360"/><img src="gifs/demo_2.gif" width="360"/>
   <img src="gifs/demo_3.gif" width="360"/><img src="gifs/demo_4.gif" width="360"/>
</div>

### Cite
If you find our work helpful or enlightening, feel free to cite our paper:
```
@INPROCEEDINGS{cao2023ariadne,
  author={Cao, Yuhong and Hou, Tianxiang and Wang, Yizhuo and Yi, Xian and Sartoretti, Guillaume},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration}, 
  year={2023},
  pages={10219-10225},
  doi={10.1109/ICRA48891.2023.10160565}}
```

### Authors
[Yuhong Cao](https://github.com/caoyuhong001)\
Tianxiang Hou\
[Yizhuo Wang](https://github.com/wyzh98)\
Xian Yi\
[Guillaume Sartoretti](https://github.com/gsartoretti)
## Online exploration feedback (2026-10-23)

Token21 now uses `new_free_cell_count`: the number of belief-map cells that
change from UNKNOWN to FREE during the latest executed action's sensor update.
The raw count is used directly, without division by map area or any scale.
Occupied discoveries and repeated observations of known free cells do not count.
This replaces the old coverage delta in the 21D token, replay, online cache,
and recovery detector. Token width remains 21 and `map_delta` still controls
this feedback component. Ground-truth coverage remains evaluation/logging only.

Recovery compares the mean raw count over the existing 8-step window with
`LOCAL_REVISIT_NEW_FREE_CELL_THRESHOLD=1.0` (initial threshold, to be calibrated).
The old 0.001 coverage threshold has no map-independent conversion to cell counts.
Position repetition and all other recovery conditions remain in place.

New checkpoints carry `exploration_progress_signal=new_free_cell_count_v1` and
layout `local_motion_target_reward_new_free_cells_v2`. Training resume and the
Token21 evaluator reject legacy coverage-feedback checkpoints instead of silently
changing their inputs. Existing historical results/checkpoints are preserved;
new training and evaluation must use separate output directories. Other model
repositories keep their existing feedback semantics in the shared evaluator.
