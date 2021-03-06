# from django.shortcuts import render
from django.http import JsonResponse

from anomaly.metrics.krum import Krum
from anomaly.metrics.zeno import Zeno
from anomaly.metrics.fools import Fools
from anomaly.metrics.auror import Auror
from anomaly.metrics.sniper import Sniper
from anomaly.metrics.pca import Pca
from backend.rfile import RFile

from backend.settings import JSON_PATH



# Create your views here.

def krum(request):

    rfile = RFile(JSON_PATH)

    k = int(request.GET.get('k', -1))
    round = int(request.GET.get('round', -1))
    layers = rfile.get_layer(request.GET.get('layers', -1))

    result = rfile.get_grad(JSON_PATH, layers, round)
    round = result['round']
    gradients = result['data']
    krum_obj = Krum(k)
    return JsonResponse({'round': round, 'data': krum_obj.get_score(gradients)}, safe=False)

# def geo_med(request):
#
#     layers = get_layer(request.GET.get('layers', -1))
#
#     gradients = get_grad('/Users/zhangtianye/Documents/FD/Femnist/test/', layers)
#     med_obj = Med(gradients)
#     return JsonResponse(med_obj.gradients['0'],safe=False)

def fools(request):

    rfile = RFile(JSON_PATH)

    k = int(request.GET.get('k', -1))
    round = int(request.GET.get('round', -1))
    layers = rfile.get_layer(request.GET.get('layers', -1))

    result = rfile.get_grad(JSON_PATH, layers, round)
    round = result['round']
    gradients = result['data']

    fools_obj = Fools(k)
    return JsonResponse({'round': round, 'data': fools_obj.score(gradients)}, safe=False)

def zeno(request):

    rfile = RFile(JSON_PATH)

    # suggest p = 100
    p = float(request.GET.get('p', -1))
    round = int(request.GET.get('round', -1))

    latest_result = rfile.get_perf(JSON_PATH, round, 'train')
    latest_round = latest_result['round']
    latest_perf = latest_result['data']
    former_perf = rfile.get_perf(JSON_PATH, latest_round - 1, 'train')['data']

    latest_grad = rfile.get_grad(JSON_PATH, ['dense'], latest_round)['data']
    former_grad = rfile.get_grad(JSON_PATH, ['dense'], latest_round - 1)['data']

    zeno_obj = Zeno(p)

    return JsonResponse({'round': latest_round, 'data': zeno_obj.score(latest_perf, former_perf, latest_grad, former_grad)}, safe=False)



def auror(request):

    rfile = RFile(JSON_PATH)

    k = int(request.GET.get('k', -1))
    round = int(request.GET.get('round', -1))
    layers = rfile.get_layer(request.GET.get('layers', -1))

    result = rfile.get_grad(JSON_PATH, layers, round)
    round = result['round']
    gradients = result['data']

    auror_obj = Auror(k)
    return JsonResponse({'round': round, 'data': auror_obj.score(gradients)}, safe=False)

def sniper(request):

    rfile = RFile(JSON_PATH)

    p = float(request.GET.get('p', -1))
    round = int(request.GET.get('round', -1))
    layers = rfile.get_layer(request.GET.get('layers', -1))

    result = rfile.get_grad(JSON_PATH, layers, round)
    round = result['round']
    gradients = result['data']

    sniper_obj = Sniper(gradients, p)
    return JsonResponse({'round': round, 'data': sniper_obj.score()}, safe=False)

# def dagmm(request):
#
#     layers = get_layer(request.GET.get('layers', -1))
#     gradients = get_grad(JSON_PATH, layers, -1)
#
#     dagmm_obj = DAGMM(gradients)
#     dagmm_obj.score()
#     return JsonResponse(['test'], safe=False)

def pca(request):

    rfile = RFile(JSON_PATH)

    k = int(request.GET.get('k', -1))
    round = int(request.GET.get('round', -1))
    layers = rfile.get_layer(request.GET.get('layers', -1))

    result = rfile.get_grad(JSON_PATH, layers, round)
    round = result['round']
    gradients = result['data']

    pca_ojb = Pca(k)
    return JsonResponse({'round': round, 'data': pca_ojb.score(gradients)}, safe=False)
