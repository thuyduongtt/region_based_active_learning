# DS_NAME = 'gland'
# IM_WIDTH = 480
# IM_HEIGHT = 480
# IM_PAD_WIDTH = 784
# IM_PAD_HEIGHT = 528
# IM_CHANNEL = 3
# INVERTED = False
# N_UNLABELED = 85
# N_SELECT = 5
# RANDOM_SELECT = False
# BATCH_SIZE = 5
# training_data_path = "DATA/Data/glanddata.npy"
# val_data_path = None
# test_data_path = "DATA/Data/glanddata_testb.npy"

DS_NAME = 'QB_128'
IM_WIDTH = 128
IM_HEIGHT = 123
IM_PAD_WIDTH = 128
IM_PAD_HEIGHT = 123
IM_CHANNEL = 6
INVERTED = True
N_UNLABELED = 48
N_SELECT = 5
RANDOM_SELECT = True
BATCH_SIZE = 5
training_data_path = "DATA/Data/" + DS_NAME + "_train.npy"
val_data_path = "DATA/Data/" + DS_NAME + "_test.npy"
test_data_path = "DATA/Data/" + DS_NAME + "_test_benign.npy"

MAX_RUN_COUNT = 1
OUTPUT_DIR = 'output'
INIT_N_EPOCH = 20
MAX_N_EPOCH = 1300
VAL_STEP = 10
