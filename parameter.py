import os

# saving path
USE_MAMBA = True
FOLDER_NAME = 'parallel_ariadne_mamba_optimized'
RUN_FOLDER_NAME = (
    f'{FOLDER_NAME}_mamba' if USE_MAMBA else FOLDER_NAME
)
model_path = f'model/{RUN_FOLDER_NAME}'
train_path = f'train/{RUN_FOLDER_NAME}'
gifs_path = f'gifs/{RUN_FOLDER_NAME}'

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
BATCH_SIZE = 8
LR = 1e-5
GAMMA = 1
# Single-GPU profile: four CPU-heavy rollout actors share one CUDA device with
# the learner, matching the efficient small-batch execution path.
GPU_COUNT = 1
NUM_META_AGENT = 4
RUNNER_GPU_FRACTION = 0.25
# Enough shared memory for four in-flight trajectories without disk spilling.
RAY_OBJECT_STORE_MEMORY = 3 * 1024 ** 3

# temporal sequence data collection
COLLECT_TRAJECTORY = True
TEMPORAL_SEQUENCE_LENGTH = 16
TEMPORAL_TOKEN_DIM = 128
ACTION_EMBEDDING_DIM = 16

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
USE_GPU = False  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
# Batch 32 runs directly on the single visible GPU.
ENABLE_MULTI_GPU_DATA_PARALLEL = False

# Ada Lovelace GPUs accelerate float32 matrix multiplications with TF32 while
# retaining float32 storage and the existing SAC numerical behaviour.
ENABLE_TF32 = True
