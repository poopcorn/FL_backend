# from django.shortcuts import render
from django.http import JsonResponse

from backend.settings import JSON_PATH
from anomaly.metrics.krum import Krum
import json




# Create your views here.

def krum(request):

    k = int(request.GET.get('k', -1))
    layers = request.GET.get('layers', ['conv1', 'conv2', 'dense'])

    # str layers 2 list
    layers = layers[1: len(layers) - 1]
    layers = layers.split(',')
    for i in range(len(layers)):
        layers[i] = layers[i][1: len(layers[i]) - 1]

    file = open('/Users/zhangtianye/Documents/FD/Femnist/test/' + 'gradients.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    length = len(data)
    krum_obj = Krum(k)
    return JsonResponse(krum_obj.get_score(data[length - 1],layers),safe=False)
