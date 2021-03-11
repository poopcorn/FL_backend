import numpy as np
from sklearn.manifold import TSNE
from pyinform import mutualinfo
from heatmap import getOneRoundFromFile
from feature import getRoundGrad
from backend.settings import DEFAULT_CLIENT_NUM


def multiple_information(start, end ,layer, filter):
    # get all metric data
    allMetrics = [getOneRoundFromFile(round, layer) for round in range(start, end + 1)]
    metricNum = sum(filter)
    roundNum = end - start + 1
    clientMetrics = np.zeros((DEFAULT_CLIENT_NUM, metricNum * roundNum))
    for clientId in range(DEFAULT_CLIENT_NUM):
        for roundIdx in range(roundNum):
            cnt = 0
            for metircId, f in enumerate(filter):
                if f == 0:
                    continue
                clientMetrics[clientId][roundIdx * metricNum + cnt] = allMetrics[roundIdx][metircId][str(clientId)]
                cnt += 1
    # calculate Mutiple Information
    multipleInfo = np.zeros((DEFAULT_CLIENT_NUM, DEFAULT_CLIENT_NUM), dtype=np.float32)
    for i in range(DEFAULT_CLIENT_NUM):
        for j in range(i + 1, DEFAULT_CLIENT_NUM):
            multipleInfo[j][i] = \
            multipleInfo[i][j] = \
                mutualinfo.mutual_info(clientMetrics[i], clientMetrics[j])
    return multipleInfo.tolist()


tsne = TSNE(n_components=2)
def get_tsne(start, end, layer):
    # get all gradient data
    allFeatureMap = [getRoundGrad(round) for round in range(start, end + 1)]
    curFeature = [v['cur'] for v in allFeatureMap]
    avgFeature = [v['avg'] for v in allFeatureMap]

    # translate into numpy
    conv_shape = np.array(curFeature[0][0][layer]).flatten().shape[0]
    roundNum = end - start + 1
    # ((clientNum + avg * roundNum), featuremap Flatten shape)
    shape = ((DEFAULT_CLIENT_NUM + 1) * roundNum, conv_shape)
    tsne_X = np.zeros(shape)
    for i in range(DEFAULT_CLIENT_NUM + 1):
        for roundIdx in range(roundNum):
            idx = i * roundNum + roundIdx
            if i == DEFAULT_CLIENT_NUM:
                # avg data
                tsne_X[idx] = np.array(avgFeature[roundIdx][layer]).flatten()
            else:
                # client data
                tsne_X[idx] = np.array(curFeature[roundIdx][i][layer]).flatten()
    tsneRes = tsne.fit_transform(tsne_X)

    # transfer to client postion and avg positon, length = DEFAULT_CLIENT_NUM + 1
    position = []
    diff = []
    avg_offset = DEFAULT_CLIENT_NUM  * roundNum
    for i in range(DEFAULT_CLIENT_NUM ):
        offset = i * roundNum
        position.append(tsneRes[offset: offset + roundNum].tolist())    # calculate Tsne
        diff.append(
            [np.linalg.norm(tsne_X[offset + roundIdx] - tsne_X[avg_offset + roundIdx]) for roundIdx in range(roundNum)]
        )
    return {
        'position': position,
        'avgPos': tsneRes[avg_offset: avg_offset + roundNum].tolist(),
        'diff': diff
    }

# get_tsne(100, 110, 'conv1')
# multiple_information(10, 12, 'conv1', [1,1,1,1,1,1,1,1,1])
