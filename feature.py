import os
import json
import pickle
import math

from backend.settings import JSON_PATH
from backend.settings import ROUND_EVERY_FILE
from backend.file import File

from backend.rfile import RFile


# READ RFILE

def readAvgGrad(allRound, clientNum=35, layers=['conv1', 'conv2']):
    rfile = RFile(JSON_PATH)
    gradientPath = JSON_PATH + 'avg_grad/'
    allFiles = os.listdir(gradientPath)
    savePath = 'data/avg_grad/'
    for fileName in allFiles:
        file = open(gradientPath + fileName, 'r', encoding='utf-8')
        data = json.load(file)
        file.close()
        for roundName in data:
            roundRes = {
                'conv1': rfile.extract_grad(data[str(roundName)]['conv1'])[0],
                'conv2': rfile.extract_grad(data[str(roundName)]['conv2'])[0]
            }
            print('save round: {}'.format(roundName))
            with open(savePath + 'round_{}.pkl'.format(roundName), 'wb') as fp:
                pickle.dump(roundRes, fp)
                fp.close()


def readAllGrad(allRound, clientNum=35, layers=['conv1', 'conv2']):
    rfile = RFile(JSON_PATH)
    gradientPath = JSON_PATH + 'client_grad/'
    allFiles = os.listdir(gradientPath)
    savePath = 'data/client_grad/'
    for fileName in allFiles:
        file = open(gradientPath + fileName, 'r', encoding='utf-8')
        data = json.load(file)
        file.close()
        for roundName in data:
            roundRes = []
            for clientId in range(clientNum):
                roundRes.append({
                    'clientId': clientId,
                    'conv1': rfile.extract_grad(data[str(roundName)][str(clientId)]['conv1'])[0],
                    'conv2': rfile.extract_grad(data[str(roundName)][str(clientId)]['conv2'])[0]
                })
            print('save round: {}'.format(roundName))
            with open(savePath + 'round_{}.pkl'.format(roundName), 'wb') as fp:
                pickle.dump(roundRes, fp)
                fp.close()

def getRoundGrad(round):
    lastRound = 'data/client_grad/' + 'round_{}.pkl'.format(min(1, round - 1))
    clientPath = 'data/client_grad/' + 'round_{}.pkl'.format(round)
    avgPath = 'data/avg_grad/' + 'round_{}.pkl'.format(round)
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

# readAllGrad(500)
# readAvgGrad(500)
# gradFile = 'data/avg_grad/round_{}.pkl'.format(500)
# with open(gradFile, 'rb') as fp:
#     gradRes = pickle.load(fp)
#     print("!!!!")