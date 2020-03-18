# from django.shortcuts import render
from django.http import JsonResponse
import json

from backend.settings import JSON_PATH

# JSON_PATH = '/Users/zhangtianye/Documents/FD/Femnist/test/'

from contribution.metrics.attention import Atten
from contribution.metrics.perf import Perf

# Create your views here.

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

def get_avg_grad(path, layers, round):
    file = open(path + 'avg_grad.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    data = data[round]['avg_grad']

    vec = []
    for item in layers:
        vec += data[item]
    return vec


# get the performance of a certain round
def get_perf(path, round, stage):
    file = open(path + 'performance.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    data = data[round][stage]

    return data

# get the contribution of a certain rouns
def get_con(path, round):
    file = open(path + 'contribution.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    data = data[round]['contribution']

    return data


def attention(request):

    metric = request.GET.get('metric', 'eu')
    layers = get_layer(request.GET.get('layers', -1))

    gradients = get_grad(JSON_PATH, layers, -1)
    avg_grad = get_avg_grad(JSON_PATH, layers, -2)

    atten_obj = Atten(metric)

    return JsonResponse(atten_obj.score(gradients, avg_grad), safe=False)

def perf_diff(request):

    metric = request.GET.get('metric', 'accuracy')
    contribution = get_con(JSON_PATH, -1)

    perf_obj = Perf(metric)

    return  JsonResponse(perf_obj.score(contribution), safe=False)