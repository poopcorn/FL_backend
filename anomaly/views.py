# from django.shortcuts import render
from django.http import JsonResponse
import json

from anomaly.metrics.krum import Krum
# from anomaly.metrics.med import Med
from anomaly.metrics.zeno import Zeno
from anomaly.metrics.fools import Fools
from anomaly.metrics.auror import Auror
from anomaly.metrics.sniper import Sniper

from backend.settings import JSON_PATH

# JSON_PATH = '/Users/zhangtianye/Documents/FD/Femnist/test/'

# change layers paramaters from str to list
def get_layer(str):
    if str == -1:
        layers = ['dense']
    else:
        layers = str[1: len(str) - 1]
        layers = layers.split(',')
        for i in range(len(layers)):
            layers[i] = layers[i][1: len(layers[i]) - 1]
    return layers

# determine gradients from which layers should be used
def get_grad(path, layers, round):

    file = open(path + 'gradients.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    data = data[round]['gradient']

    # get gradients of all clients
    vec = {}
    for key in data:
        vec[key] = []
        for item in layers:
            vec[key] += data[key][item]
    return vec

# get the performance of a certain round
def get_perf(path, round, stage):
    file = open(path + 'performance.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    data = data[round][stage]

    return data


# Create your views here.

def krum(request):

    k = int(request.GET.get('k', -1))
    layers = get_layer(request.GET.get('layers', -1))

    gradients = get_grad(JSON_PATH, layers, -1)
    krum_obj = Krum(k)
    return JsonResponse(krum_obj.get_score(gradients),safe=False)

# def geo_med(request):
#
#     layers = get_layer(request.GET.get('layers', -1))
#
#     gradients = get_grad('/Users/zhangtianye/Documents/FD/Femnist/test/', layers)
#     med_obj = Med(gradients)
#     return JsonResponse(med_obj.gradients['0'],safe=False)

def zeno(request):

    # suggest p = 100
    p = float(request.GET.get('p', -1))

    perf_this = get_perf(JSON_PATH, -1, 'train')
    perf_last = get_perf(JSON_PATH, -2, 'train')
    grad_this = get_grad(JSON_PATH, ['dense'], -1)
    grad_last = get_grad(JSON_PATH, ['dense'], -2)

    zeno_obj = Zeno(p)

    return JsonResponse(zeno_obj.score(perf_this, perf_last, grad_this, grad_last), safe=False)

def fools(request):

    k = int(request.GET.get('k', -1))
    layers = get_layer(request.GET.get('layers', -1))

    gradients = get_grad(JSON_PATH, layers, -1)
    fools_obj = Fools(k)
    return JsonResponse(fools_obj.score(gradients), safe=False)

def auror(request):

    k = int(request.GET.get('k', -1))
    layers = get_layer(request.GET.get('layers', -1))
    gradients = get_grad(JSON_PATH, layers, -1)

    auror_obj = Auror(k)


    return JsonResponse(auror_obj.score(gradients), safe=False)

def sniper(request):

    p = float(request.GET.get('p', -1))
    layers = get_layer(request.GET.get('layers', -1))
    gradients = get_grad(JSON_PATH, layers, -1)

    sniper_obj = Sniper(gradients, p)
    return JsonResponse(sniper_obj.score(), safe=False)
