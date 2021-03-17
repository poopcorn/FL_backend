# 本次模型和数据集对应的全局数据
# DEFAULT_MODEL = 'EasyCNN'
# DEFAULT_DATASET = 'FEMINIST'

DEFAULT_MODEL = 'LeNet'
DEFAULT_DATASET = 'DIGIT5'
# DEFAULT_DATASET = 'FEMINIST'
DATA_SAVE_FILE = 'data/{}_{}'.format(DEFAULT_MODEL, DEFAULT_DATASET)
DEFAULT_CLIENT_NUM = 35
DEFAULT_ROUND_NUM = 150
LAYERS_NANME = ['layer1', 'layer2']
ROUND_EVERY_FILE = 10

# 模型计算数据保存目录
JSON_PATH = '/home/zty_11621014/fedcare/federated/data/experiments/FEMNIST/'
JSON_PREFIX = '/home/zty_11621014/fedcare/federated/data/experiments/'