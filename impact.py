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

def get_tsne(start, end):
    # get all gradient data
    allFeatureMap = [getRoundGrad(round) for round in range(start, end + 1)]
    curFeature = [v['cur'] for v in allFeatureMap]
    avgFeature = [v['avg'] for v in allFeatureMap]

    # translate into numpy
    conv1_shape = np.array(curFeature[0][0]['conv1']).flatten().shape[0]
    conv2_shape = np.array(curFeature[0][0]['conv2']).flatten().shape[0]
    roundNum = end - start + 1
    # (clientNum, (roundNum + roundNum//avg), featuremap Flatten shape)
    tsne_X = {
        'conv1': np.zeros((DEFAULT_CLIENT_NUM, roundNum * 2, conv1_shape)),
        'conv2': np.zeros((DEFAULT_CLIENT_NUM, roundNum * 2, conv2_shape)),
    }
    client_tsne = {
        'conv1': [],
        'conv2': []
    }
    for i in range(DEFAULT_CLIENT_NUM):
        for roundIdx in range(roundNum):
            tsne_X['conv1'][i][roundIdx] = np.array(curFeature[roundIdx][i]['conv1']).flatten()
            # avg data
            tsne_X['conv1'][i][roundNum + roundIdx] = np.array(avgFeature[roundIdx]['conv1']).flatten()

            tsne_X['conv2'][i][roundIdx] = np.array(curFeature[roundIdx][i]['conv2']).flatten()
            # avg data
            tsne_X['conv2'][i][roundNum + roundIdx] = np.array(avgFeature[roundIdx]['conv2']).flatten()    
        # calculate Tsne
        client_tsne['conv1'].append(TSNE(n_components=2).fit_transform(tsne_X['conv1'][i]).tolist())
        client_tsne['conv2'].append(TSNE(n_components=2).fit_transform(tsne_X['conv2'][i]).tolist())
    return client_tsne

# get_tsne(10, 12)
# multiple_information(10, 12, 'conv1', [1,1,1,1,1,1,1,1,1])
