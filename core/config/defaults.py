from yacs.config import CfgNode as CN


_C = CN()

#########################################################################################
# General Parameters
#########################################################################################
_C.TASK = 'pixel' # pixel (segmentation & normal) vs. image (classification)
_C.DATASET = 'nyu_v2'  # or 'taskonomy'

_C.LOG_DIR = 'run'  # Tensorboard log directory
_C.SAVE_DIR = 'ckpts'

_C.CUDA = True
_C.SEED = 1

#########################################################################################
# Parameters for specific models
#########################################################################################
_C.MODEL = CN()

_C.MODEL.SINGLETASK = False
_C.MODEL.SHAREDFEATURE = False
_C.MODEL.SUPERNET = False

# Parameter for NAS
_C.MODEL.BACKBONE = 'VGG16_13_Stage'

_C.MODEL.INIT = (0.9, 0.1)

_C.MODEL.NDDR_TYPE = ''
_C.MODEL.NDDR_BN_TYPE = 'default'

_C.MODEL.ZERO_BATCH_NORM_GAMMA = False
_C.MODEL.BATCH_NORM_MOMENTUM = 0.05

_C.MODEL.NET1_CLASSES = 40
_C.MODEL.NET2_CLASSES = 3


#########################################################################################
# Parameters for Architecture Search
#########################################################################################
_C.ARCH = CN()

_C.ARCH.SEARCHSPACE = '' # Run nddr when this is empty

_C.ARCH.TRAIN_SPLIT = 0.5  # portion of the original training data to keep, with the rest being used for nas
_C.ARCH.MIXED_DATA = True

# Optimization
_C.ARCH.OPTIMIZER = ''
_C.ARCH.LR = 3e-3
_C.ARCH.WEIGHT_DECAY = 1e-3

# For Gumbel Softmax on model connections
_C.ARCH.INIT_TEMP = 1.
_C.ARCH.TEMPERATURE_POWER = 2.
_C.ARCH.TEMPERATURE_PERIOD = (0., 1.)

# For regularization
_C.ARCH.ENTROPY_REGULARIZATION = False
_C.ARCH.ENTROPY_PERIOD = (0., 1.)  # proportion of training with regularization
_C.ARCH.ENTROPY_REGULARIZATION_WEIGHT = 10.  # 10. or 50.
_C.ARCH.L1_REGULARIZATION = False
_C.ARCH.L1_OFF = False  # turn off l1 after certain period
_C.ARCH.WEIGHTED_L1 = False
_C.ARCH.L1_PERIOD = (0., 1.)
_C.ARCH.L1_REGULARIZATION_WEIGHT = 5.

# Feedforward hard vs. soft options
_C.ARCH.HARD_WEIGHT_TRAINING = True  # use gumbel trick for feedforward
_C.ARCH.HARD_ARCH_TRAINING = True  # use gumbel trick for feedforward
_C.ARCH.HARD_EVAL = True  # whether to only take most likely operation during test time
_C.ARCH.STOCHASTIC_EVAL = False  # for SNAS eval


#########################################################################################
# Parameters for Model Training
#########################################################################################
_C.TRAIN = CN()
_C.TRAIN.APEX = False

_C.TRAIN.FREEZE_BASE = False

_C.TRAIN.AUX = False
_C.TRAIN.AUX_WEIGHT = 0.4
_C.TRAIN.AUX_PERIOD = (0., 0.)
_C.TRAIN.AUX_LAYERS = []

_C.TRAIN.COLOR_JITTER = True
_C.TRAIN.RANDOM_SCALE = True
_C.TRAIN.RANDOM_MIRROR = True
_C.TRAIN.RANDOM_CROP = True
_C.TRAIN.OUTPUT_SIZE = (321, 321)

_C.TRAIN.WEIGHT_1 = 'DeepLab'
_C.TRAIN.WEIGHT_2 = 'DeepLab'

_C.TRAIN.BATCH_SIZE = 10
_C.TRAIN.STEPS = 20001
_C.TRAIN.WARMUP = 0
_C.TRAIN.LR = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 2.5e-4
_C.TRAIN.POWER = 0.9
_C.TRAIN.NDDR_FACTOR = 100.
_C.TRAIN.FC8_WEIGHT_FACTOR = 10.
_C.TRAIN.FC8_BIAS_FACTOR = 20.
_C.TRAIN.TASK2_FACTOR = 20.  # 20. for normal
_C.TRAIN.SCHEDULE = 'Poly'

_C.TRAIN.LOG_INTERVAL = 500
_C.TRAIN.EVAL_INTERVAL = 1000
_C.TRAIN.SAVE_INTERVAL = 1000
_C.TRAIN.EVAL_CKPT = True


#########################################################################################
# Parameters for Model Testing
#########################################################################################
_C.TEST = CN()

_C.TEST.RANDOM_SCALE = False
_C.TEST.RANDOM_MIRROR = False
_C.TEST.RANDOM_CROP = False

_C.TEST.BATCH_SIZE = 10

_C.TEST.CKPT_ID = 20000
