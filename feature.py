import os
import json
import pickle
import math

from const import *
from backend.file import File
from backend.rfile import RFile
from const import LAYERS_NANME



# READ RFILE
'''
    保存对应轮次中,所有client的平均梯度，包括所有层
    @params:
        prefix: 存储文件夹的路径
        allRound: 所有轮次的数量
        clientNum: client数量
        layers: string list, 哪些层要保留
'''
def saveAvgGrad(prefix, allRound, clientNum=35, layers=LAYERS_NANME):
    rfile = RFile(JSON_PATH)
    gradientPath = JSON_PATH + 'avg_grad/'
    allFiles = os.listdir(gradientPath)
    savePath = prefix + '/avg_grad/'
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    for fileName in allFiles:
        file = open(gradientPath + fileName, 'r', encoding='utf-8')
        data = json.load(file)
        file.close()
        for roundName in data:
            roundRes = {
                LAYERS_NANME[0]: rfile.extract_grad(data[str(roundName)][LAYERS_NANME[0]])[0],
                LAYERS_NANME[1]: rfile.extract_grad(data[str(roundName)][LAYERS_NANME[1]])[0]
            }
            print('save round: {}'.format(roundName))
            with open(savePath + 'round_{}.pkl'.format(roundName), 'wb') as fp:
                pickle.dump(roundRes, fp)
                fp.close()

'''
    保存对应轮次中,所有client的所有梯度，包括所有层
    @params:
        prefix: 存储文件夹的路径
        allRound: 所有轮次的数量
        clientNum: client数量
        layers: string list, 哪些层要保留
'''
def saveAllGrad(prefix, allRound, clientNum=35, layers=LAYERS_NANME):
    rfile = RFile(JSON_PATH)
    gradientPath = JSON_PATH + 'client_grad/'
    allFiles = os.listdir(gradientPath)
    savePath = prefix + '/client_grad/'
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    for fileName in allFiles:
        file = open(gradientPath + fileName, 'r', encoding='utf-8')
        data = json.load(file)
        file.close()
        for roundName in data:
            roundRes = []
            filePath = savePath + 'round_{}.pkl'.format(roundName)
            if os.path.exists(filePath):
                print('{} has been saved'.format(filePath))
                continue
            for clientId in range(clientNum):
                roundRes.append({
                    'clientId': clientId,
                    LAYERS_NANME[0]: rfile.extract_grad(data[str(roundName)][str(clientId)][LAYERS_NANME[0]])[0],
                    LAYERS_NANME[1]: rfile.extract_grad(data[str(roundName)][str(clientId)][LAYERS_NANME[1]])[0]
                })
            print('save round: {}'.format(roundName))
            with open(savePath + 'round_{}.pkl'.format(roundName), 'wb') as fp:
                pickle.dump(roundRes, fp)
                fp.close()

def getRoundGrad(round):
    lastRound = '{}/client_grad/round_{}.pkl'.format(DATA_SAVE_FILE, min(1, round - 1))
    clientPath = '{}/client_grad/round_{}.pkl'.format(DATA_SAVE_FILE,round)
    avgPath = '{}/avg_grad/round_{}.pkl'.format(DATA_SAVE_FILE, round)
    with open(clientPath, 'rb') as fp:
        curRound = pickle.load(fp)
        fp.close()
    
    with open(lastRound, 'rb') as fp:
        lastRound = pickle.load(fp)
        fp.close()
    
    with open(avgPath, 'rb') as fp:
        avgRes = pickle.load(fp)
        fp.close()
    return {
        'last': lastRound,
        'cur': curRound,
        'avg': avgRes
    }

# getRoundGrad(20)