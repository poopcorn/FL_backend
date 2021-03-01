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


rfile = RFile(JSON_PATH)
krum_obj = Krum(-1)
fools_obj = Fools(-1)
auror_obj = Auror(5)
zeno_obj = Zeno(100)
pca_obj = Pca(5)
atten_eu_obj = Atten('eu')
atten_cos_obj = Atten('cos')
perf_a_obj = Perf('accuracy')
perf_l_obj = Perf('loss')
def getOneRound(round, layers):
    # Anomaly Metrics
    result = rfile.get_grad(JSON_PATH, layers, round)
    after_round = result['round']
    gradients = result['data']
    avg_grad = rfile.get_avg_grad(JSON_PATH, layers, max(1, after_round - 1))['data']
    krum_res = krum_obj.get_score(gradients)
    fools_res = fools_obj.score(gradients)
    auror_res = auror_obj.score(gradients)
    sniper_res = Sniper(gradients, 0.8).score()
    pca_res = pca_obj.score(gradients)

    latest_result = rfile.get_perf(JSON_PATH, round, 'train')
    latest_round = latest_result['round']
    latest_perf = latest_result['data']
    former_perf = rfile.get_perf(JSON_PATH, max(1, latest_round - 1), 'train')['data']

    latest_grad = rfile.get_grad(JSON_PATH, ['dense'], latest_round)['data']
    former_grad = rfile.get_grad(JSON_PATH, ['dense'], max(1, latest_round - 1))['data']
    zeno_res = zeno_obj.score(latest_perf, former_perf, latest_grad, former_grad)

    # Contribution Metrics
    atten_eu_res = atten_eu_obj.score(gradients, avg_grad)
    atten_cos_res = atten_cos_obj.score(gradients, avg_grad)
    no_layer_result = rfile.get_con(JSON_PATH, round)
    contribution = no_layer_result['data']
    perf_a_res = perf_a_obj.score(contribution)
    perf_l_res = perf_l_obj.score(contribution)
    # res = {
    #     'Krum': krum_res,
    #     'FoolsGold': fools_res,
    #     'Zeno': zeno_res,
    #     'Auror': auror_res,
    #     'Sniper': sniper_res,
    #     'Pca': pca_res,
    #     'eu': atten_eu_res,
    #     'cos': atten_cos_res,
    #     'accuracy': perf_a_res,
    #     'loss': perf_l_res
    # }
    res = [krum_res, fools_res, zeno_res, auror_res, sniper_res, pca_res, atten_eu_res, atten_cos_res, perf_a_res, perf_l_res]
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
