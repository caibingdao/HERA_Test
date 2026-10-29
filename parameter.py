import os
from temporal_pose import pose_config


def _environment_flag(name, default):
    """Read an optional strict boolean environment override."""
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(
        f'{name} must be one of 1/0, true/false, yes/no, or on/off; '
        f'got {value!r}'
    )


# saving path
USE_MAMBA = True
FOLDER_NAME = os.environ.get(
    'ARIADNE_RUN_NAME', 'parallel_ariadne_mamba_3_1'
)
RUN_FOLDER_NAME = (
    FOLDER_NAME if USE_MAMBA else f'{FOLDER_NAME}_baseline'
)
model_path = f'model/{RUN_FOLDER_NAME}'
train_path = f'train/{RUN_FOLDER_NAME}'
_gif_model_type = (
    'recovery_on'
    if _environment_flag('ARIADNE_USE_RECOVERY', True)
    else 'recovery_off'
)
gifs_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'gifs',
    _gif_model_type,
    RUN_FOLDER_NAME,
)
if os.environ.get('ARIADNE_OUTPUT_ROOT'):
    model_path = os.path.join(os.environ['ARIADNE_OUTPUT_ROOT'], 'model')
    train_path = os.path.join(os.environ['ARIADNE_OUTPUT_ROOT'], 'train')
gifs_path = os.environ.get('ARIADNE_GIFS_PATH', gifs_path)

# save training data
SUMMARY_WINDOW = 32  # how many training steps before writing data to tensorboard
LOAD_MODEL = os.environ.get('ARIADNE_LOAD_MODEL', '0') == '1'
SAVE_IMG_GAP = 100  # how many episodes before saving a gif
# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 4.0  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers

# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# sensor and utility range
SENSOR_RANGE = 16  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 2  # ignore the utility if observable frontiers are less than this value

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 3000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 32
LEARNER_UPDATES_PER_TRAIN = 8
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 5  # five rollout workers on one dedicated rollout GPU
RUNNER_GPU_FRACTION = 0.20
RAY_ROLLOUT_GPUS = 1  # visible GPU 0 collects rollouts
# The original two-GPU layout uses visible GPU 1 for the learner. Ablation
# jobs expose exactly one physical GPU and override this to visible GPU 0.
LEARNER_GPU_INDEX = int(
    os.environ.get('ARIADNE_LEARNER_GPU_INDEX', '1')
)
RAY_OBJECT_STORE_MEMORY = 3 * 1024 ** 3
RANDOM_SEED = int(os.environ.get('ARIADNE_RANDOM_SEED', '4777'))

# Optional leakage-safe map manifest. Training accepts only the manifest's
# train side; an unset path preserves the legacy behavior of using all maps/.
MAP_SPLIT_FILE = os.environ.get('ARIADNE_MAP_SPLIT_FILE')
MAP_SPLIT_NAME = os.environ.get('ARIADNE_MAP_SPLIT_NAME', 'train')
if MAP_SPLIT_FILE is not None and MAP_SPLIT_NAME != 'train':
    raise ValueError(
        'ARIADNE_MAP_SPLIT_NAME must be train during policy training'
    )

# temporal sequence data collection
COLLECT_TRAJECTORY = True
TEMPORAL_SEQUENCE_LENGTH = 16
TEMPORAL_TOKEN_DIM = 128
ACTION_EMBEDDING_DIM = 16
# This variant uses egocentric transition motion in each history token.
TEMPORAL_POSE_ENCODING = os.environ.get(
    'ARIADNE_TEMPORAL_POSE_ENCODING', 'delta_local'
)
TEMPORAL_DELTA_POSITION_SCALE = float(os.environ.get(
    'ARIADNE_TEMPORAL_DELTA_POSITION_SCALE', str(NODE_RESOLUTION)
))
TEMPORAL_TOKEN_VERSION = pose_config(
    TEMPORAL_POSE_ENCODING, TEMPORAL_DELTA_POSITION_SCALE
)['temporal_token_version']
ALLOW_TEMPORAL_POSE_OVERRIDE = _environment_flag('ARIADNE_ALLOW_TEMPORAL_POSE_OVERRIDE', False)

# History-token ablation switches for the 21D raw layout. Disabled groups
# are zeroed after normalization and before projection, so tensor widths stay
# fixed.
#
# features: enhanced target-node feature (projected to 16D)
# behavior_reward: local dx/dy/turn angle + normalized reward
# map_delta: raw count of newly observed free cells (no normalization)
USE_HISTORY_FEATURES_IN_TOKEN = _environment_flag(
    'ARIADNE_TOKEN_FEATURES', True
)
USE_HISTORY_BEHAVIOR_REWARD_IN_TOKEN = _environment_flag(
    'ARIADNE_TOKEN_BEHAVIOR_REWARD', True
)
USE_HISTORY_MAP_DELTA_IN_TOKEN = _environment_flag(
    'ARIADNE_TOKEN_MAP_DELTA', True
)
# Fine-grained pose ablation: only zero local dx/dy/dtheta.
USE_HISTORY_MOTION_DELTA_IN_TOKEN = _environment_flag(
    'ARIADNE_TOKEN_MOTION_DELTA', True
)

# reversible phase-two history optimizations
USE_ROLLOUT_HISTORY_CACHE = True
USE_LEARNER_HISTORY_DEDUP = False

# network parameters
NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128

# Mamba temporal branch and concat-MLP fusion
MAMBA_SEQ_LEN = TEMPORAL_SEQUENCE_LENGTH
MAMBA_D_MODEL = TEMPORAL_TOKEN_DIM
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
FUSION_HIDDEN_DIM = EMBEDDING_DIM

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes, fixed
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = True  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
ENABLE_MULTI_GPU_DATA_PARALLEL = False
ENABLE_DISTRIBUTED_LEARNER = False
DISTRIBUTED_BACKEND = 'nccl'

# learner performance experiments (all disabled by default)
ENABLE_TRAIN_PROFILING = True
PROFILE_WARMUP_UPDATES = 10
PROFILE_ACTIVE_UPDATES = 50
PROFILE_LOG_INTERVAL = 10

USE_BF16 = True
CHECK_TRAIN_FINITE = False
USE_HISTORY_CHECKPOINT = False
HISTORY_GRAPH_BATCH_SIZE = 32
SKIP_PADDED_HISTORY_GRAPHS = True
TRIM_HISTORY_GRAPH_NODES = True
REUSE_CURRENT_Q_FORWARD = False
USE_NON_BLOCKING_TRANSFER = False
USE_UNION_HISTORY_COLLATE = True

# Loop Recovery V4.0: regional revisit plus point-repeat detection.
# Set this to False to preserve the original policy behavior exactly.
USE_LOCAL_REVISIT_RECOVERY = _environment_flag(
    'ARIADNE_USE_RECOVERY', True
)
LOCAL_REVISIT_HISTORY_LENGTH = 10
LOCAL_REVISIT_GRAPH_RADIUS = 2 * NODE_RESOLUTION
LOCAL_REVISIT_MIN_COUNT = 7
LOCAL_REVISIT_SCORE_THRESHOLD = (
    LOCAL_REVISIT_MIN_COUNT / LOCAL_REVISIT_HISTORY_LENGTH
)
# Compatibility/diagnostic value only; Recovery V2.1 does not exit on score.
LOCAL_REVISIT_EXIT_THRESHOLD = 0.5
LOCAL_REVISIT_PROGRESS_WINDOW = 8
# Initial raw-count threshold: low progress means <1 new free cell/step
# averaged over the progress window. No map-area normalization.
LOCAL_REVISIT_NEW_FREE_CELL_THRESHOLD = 1.0
LOCAL_REVISIT_TRIGGER_STEPS = int(os.environ.get('ARIADNE_RECOVERY_TRIGGER_STEPS', '2'))

# Point-repeat evidence is counted only inside the mapped 10-step history.
# It owns a separate streak counter so regional and point evidence cannot
# alternate across steps and accidentally satisfy one shared trigger gate.
USE_POINT_REPEAT_DETECTION = True
POINT_REPEAT_MAX_COUNT = 3
POINT_REPEAT_TOP_K = 3
POINT_REPEAT_TOP_K_COUNT = 8
POINT_REPEAT_TRIGGER_STEPS = int(os.environ.get(
    'ARIADNE_POINT_REPEAT_TRIGGER_STEPS', str(LOCAL_REVISIT_TRIGGER_STEPS)
))
if LOCAL_REVISIT_TRIGGER_STEPS < 1 or POINT_REPEAT_TRIGGER_STEPS < 1:
    raise ValueError('Recovery trigger streaks must be positive')
LOCAL_REVISIT_DEBUG_VISUALIZATION = False
RECOVERY_GOAL_TOLERANCE = 0.5 * NODE_RESOLUTION
RECOVERY_MAX_STEPS = 32
RECOVERY_PATH_FAILURE_LIMIT = 3

# Frontier-driven target selection is retained from V3.0. Disable this switch
# to recover the V2.1 nearest-node target selection for ablation experiments.
USE_FRONTIER_ESCAPE = True
ESCAPE_MIN_DISTANCE = 12.0
ESCAPE_MIN_LOOP_DISTANCE = 8.0
ESCAPE_DISTANCE_EPS = 1.0

# Recovery diagnostics. These do not affect triggering or action selection.
ENABLE_RECOVERY_VIS = True
SAVE_RECOVERY_EVENT_LOG = True
RECOVERY_EVENT_LOG_PATH = f'{gifs_path}/recovery_events'
SAVE_TRIGGERED_RECOVERY_ARTIFACTS = _environment_flag('ARIADNE_SAVE_TRIGGERED_RECOVERY_ARTIFACTS', True)
TRIGGERED_RECOVERY_ARTIFACT_PATH = (
    f'{gifs_path}/loop_recovery_v4_triggered'
)
TRIGGERED_RECOVERY_GIF_DURATION = 0.5
RECOVERY_TRIGGER_COLOR = 'red'
RECOVERY_TRIGGER_MARKER = 'x'
RECOVERY_TRIGGER_MARKER_SIZE = 80
RECOVERY_TRIGGER_LINE_WIDTH = 2
RECOVERY_TRIGGER_LABEL_FONT_SIZE = 9
RECOVERY_TRIGGER_LABEL_OFFSET_CELLS = (2.0, -2.0)
RECOVERY_PATH_COLOR = 'gold'
RECOVERY_PATH_LINE_WIDTH = 3
RECOVERY_TARGET_COLOR = 'lime'
RECOVERY_TARGET_MARKER = '*'
RECOVERY_TARGET_MARKER_SIZE = 110

# Independent token reward mask; SAC learning rewards remain unchanged.
USE_HISTORY_REWARD_IN_TOKEN = _environment_flag("ARIADNE_TOKEN_REWARD", True)

# Frontier-count change remains recorded for diagnostics/recovery but is not
# part of this variant's temporal token.
USE_HISTORY_FRONTIER_DELTA_IN_TOKEN = False

# Optional independent target-node feature override; legacy behavior follows features.
USE_HISTORY_ACTION_FEATURES_IN_TOKEN = _environment_flag(
    "ARIADNE_TOKEN_ACTION_FEATURES", USE_HISTORY_FEATURES_IN_TOKEN
)
