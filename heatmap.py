import os
import json
import pickle
import math

from backend.settings import JSON_PATH
from backend.settings import ROUND_EVERY_FILE
from backend.file import File

from anomaly.metrics.krum import Krum
from anomaly.metrics.zeno import Zeno
from anomaly.metrics.fools import Fools
from anomaly.metrics.auror import Auror
from anomaly.metrics.sniper import Sniper
from anomaly.metrics.pca import Pca
from backend.rfile import RFile

from contribution.metrics.attention import Atten
from contribution.metrics.perf import Perf


# READ RFILE
rfile = RFile(JSON_PATH)
sniper_obj = Sniper(0.8)
fools_obj = Fools(-1)
auror_obj = Auror(1)
zeno_obj = Zeno(100)
pca_obj = Pca(5)
atten_eu_obj = Atten('eu')
atten_cos_obj = Atten('cos')
perf_a_obj = Perf('accuracy')
perf_l_obj = Perf('loss')

# LOAD CLIENT DATA
conv1AllRoundFile = 'data/{}_{}.pkl'.format(500, 'conv1')
allRoundRes = {
    'conv1': [],
    'conv2': []
}
if os.path.exists(conv1AllRoundFile):
    with open(conv1AllRoundFile, 'rb') as fp:
        allRoundRes['conv1'] = pickle.load(fp)
        fp.close()

def getOneRoundFromFile(curRound, layer):
    if curRound < 500 and curRound > 2:
        if layer == 'conv1':
            return allRoundRes['conv1'][curRound]
        elif layer == 'conv2':
            return allRoundRes['conv2'][curRound]
    return getOneRound(curRound, layer)


def getOneRound(round, layers):
    # Anomaly Metrics
    result = rfile.get_grad(JSON_PATH, layers, round)
    after_round = result['round']
    gradients = result['data']
    data = rfile.reshape_grad(gradients)

    # krum --> fools
    krum_scores = []
    for i in range(len(data)):
        krum_scores.append(fools_obj.score(data[i]))
    krum_res = rfile.avg_score(krum_scores)
 
    # auror
    auror_scores = []
    for i in range(len(data)):
        auror_scores.append(auror_obj.score(data[i]))
    auror_res = rfile.avg_score(auror_scores)

    # sniper
    sniper_scores = []
    for i in range(len(data)):
        sniper_scores.append(sniper_obj.score(data[i]))
    sniper_res = rfile.avg_score(sniper_scores)

    # pca
    pca_scores = []
    for i in range(len(data)):
        pca_scores.append(pca_obj.score(data[i]))
    pca_res = rfile.avg_score(pca_scores)

    # zeno
    latest_result = rfile.get_perf(JSON_PATH, round, 'train')
    latest_round = latest_result['round']
    latest_perf = latest_result['data']
    former_perf = rfile.get_perf(JSON_PATH, latest_round - 1, 'train')['data']

    latest_grad = rfile.get_grad(JSON_PATH, layers, latest_round)['data']
    reshape_latest_grad = rfile.reshape_grad(latest_grad)
    former_grad = rfile.get_grad(JSON_PATH, layers, latest_round - 1)['data']
    reshape_former_grad = rfile.reshape_grad(former_grad)

    zeno_scores = []
    for i in range(len(reshape_latest_grad)):
        zeno_scores.append(zeno_obj.score(latest_perf, former_perf, reshape_latest_grad[i], reshape_former_grad[i]))
    zeno_res = rfile.avg_score(zeno_scores)

    # Contribution Metrics
    avg_grad = rfile.get_avg_grad(JSON_PATH, layers, round - 1)['data']
    avg_data = rfile.reshape_avg_grad(avg_grad)

    # attention
    atten_eu_scores = []
    for i in range(len(data)):
        atten_eu_scores.append(atten_eu_obj.score(data[i], avg_data[i]))
    atten_eu_res = rfile.avg_score(atten_eu_scores)

    atten_cos_scores = []
    for i in range(len(data)):
        atten_cos_scores.append(atten_cos_obj.score(data[i], avg_data[i]))
    atten_cos_res = rfile.avg_score(atten_cos_scores)

    # perf diff
    no_layer_result = rfile.get_con(JSON_PATH, round)
    contribution = no_layer_result['data']
    perf_a_res = perf_a_obj.score(contribution)
    perf_l_res = perf_l_obj.score(contribution)
    # res = {
    #     'Krum': krum_res,
    #     'Zeno': zeno_res,
    #     'Auror': auror_res,
    #     'Sniper': sniper_res,
    #     'Pca': pca_res,
    #     'eu': atten_eu_res,
    #     'cos': atten_cos_res,
    #     'accuracy': perf_a_res,
    #     'loss': perf_l_res
    # }
    res = [krum_res, zeno_res, auror_res, sniper_res, pca_res, atten_eu_res, atten_cos_res, perf_a_res, perf_l_res]
    return res

def get_all_round():
    res = [{
        'anomaly': [[] for i in range(6)],
        'contribution': [[] for i in range(4)]
    } for client in range(35)]
    layers = ['dense']
    for round in range(1, 500):
        print(round)
        oneRes = getOneRound(round, layers)
        for client in range(35):
            for i in range(6):
                tmp = oneRes[i][str(client)]
                res[client]['anomaly'][i].append(0 if math.isnan(tmp) else tmp)
            for i in range(4):
                tmp = oneRes[i + 6][str(client)]
                res[client]['contribution'][i].append(0 if math.isnan(tmp) else tmp)

    with open('data/dense_metrics.json', 'w') as fp:
        json.dump(res, fp)
    with open('data/dense_metrics.pkl', 'wb') as fp:
        pickle.dump(res, fp)

# getOneRound(100, 'conv1')

def saveOneRound(roundNum, layer, name):
    path = 'data/{}_{}_{}.pkl'.format(roundNum, layer, name)
    if os.path.exists(path):
        print('file {path} has already saved!')
        return
    res = [[], []]
    for i in range(2, roundNum):
        print(i)
        res.append(getOneRound(i, layer))
    with open(path, 'wb') as fp:
        pickle.dump(res, fp)
        fp.close()

# saveOneRound(500, 'conv1', 'auror=1')
# allRoundFile = 'data/{}_{}.pkl'.format(500, 'conv1')
# with open(allRoundFile, 'rb') as fp:
#     allRoundRes = pickle.load(fp)
#     print("!!!!")
