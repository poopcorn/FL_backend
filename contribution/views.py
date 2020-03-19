# from django.shortcuts import render
from django.http import JsonResponse
import json

from backend.settings import JSON_PATH


from contribution.metrics.attention import Atten
from contribution.metrics.perf import Perf
from backend.rfile import RFile

# Create your views here.

def attention(request):

    rfile = RFile(JSON_PATH)

    metric = request.GET.get('metric', 'eu')
    round = int(request.GET.get('round', -1))
    layers = rfile.get_layer(request.GET.get('layers', -1))

    result = rfile.get_grad(JSON_PATH, layers, round)
    round = result['round']
    gradients = result['data']

    avg_grad = rfile.get_avg_grad(JSON_PATH, layers, round - 1)['data']

    atten_obj = Atten(metric)

    return JsonResponse({'round': round, 'data': atten_obj.score(gradients, avg_grad)}, safe=False)

def perf_diff(request):

    rfile = RFile(JSON_PATH)

    metric = request.GET.get('metric', 'accuracy')
    round = int(request.GET.get('round', -1))

    result = rfile.get_con(JSON_PATH, round)
    round = result['round']
    contribution = result['data']

    perf_obj = Perf(metric)

    return  JsonResponse({'round': round, 'data': perf_obj.score(contribution)}, safe=False)