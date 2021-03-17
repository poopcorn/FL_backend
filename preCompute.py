import os
# from const import *
from heatmap import saveOneRound
from feature import saveAllGrad, saveAvgGrad

# if not os.path.exists(DATA_SAVE_FILE):
#     os.mkdir(DATA_SAVE_FILE)
#
# roundNum = DEFAULT_ROUND_NUM

# 存储metric数据
# print('Begin compute metric data.......')
# for layer in LAYERS_NANME:
#     if os.path.exists('{}/{}_{}.pkl'.format(DATA_SAVE_FILE, roundNum, layer)):
#         continue
#     saveOneRound(roundNum, layer, DATA_SAVE_FILE)
# print('Finish compute metric data!')

# 存储梯度数据
# print('Begin compute gradient data......')
# saveAllGrad(DATA_SAVE_FILE, roundNum, DEFAULT_CLIENT_NUM, LAYERS_NANME)
# saveAvgGrad(DATA_SAVE_FILE, roundNum, DEFAULT_CLIENT_NUM, LAYERS_NANME)
# print('Finish compute gradient data!')

from backend.rfile import RFile
from anomaly.metrics.sniper import Sniper
import const


rfile = RFile(const.JSON_PATH)

for round in range(1, 149):
    p = 0.8
    layers = 'layer1'

    result = rfile.get_grad(const.JSON_PATH, layers, round)
    gradients = result['data']
    data = rfile.reshape_grad(gradients)

    sniper_obj = Sniper(p)

    scores = sniper_obj.score(data[0])
    print(round, scores)

