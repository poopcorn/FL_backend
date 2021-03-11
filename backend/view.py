from django.http import HttpResponse
from django.http import JsonResponse
import json
import pickle as pkl

from backend.settings import JSON_PATH
from backend.settings import ROUND_EVERY_FILE
from backend.file import File
from heatmap import getOneRound, getOneRoundFromFile, rfile
from impact import multiple_information, get_tsne
from feature import getRoundGrad


def performance(request):

    round = int(request.GET.get('round', -1))
    num = int(request.GET.get('number', 1))

    file = open(JSON_PATH + 'performance.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()

    if round == -1:
        performance = data
    elif num == 1:
        performance = data[str(round)]
    else:
        performance = {}
        for i in range(round - num + 1, round + 1):
            performance[str(i)] = data[str(i)]

    return JsonResponse(performance, safe=False)

def get_grad_by_round(request):
    round = int(request.GET.get('round', -1))
    return JsonResponse(getRoundGrad(round), safe=False)

def client_grad(request):

    round = int(request.GET.get('round', -1))

    if round == -1:
        file_obj = File(JSON_PATH + 'client_grad', 'gradients_')
        filename = file_obj.latest_file(ROUND_EVERY_FILE)
    else:
        filename = 'gradients_' + \
                   str((round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE) + '_' + \
                   str((round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE + ROUND_EVERY_FILE - 1) + '.json'

    file = open(JSON_PATH + 'client_grad/' + filename, 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    
    if round == -1:
        return JsonResponse({'round': int(list(data.keys())[-1]), 'data': data[list(data.keys())[-1]]}, safe=False)
    else:
        return JsonResponse(data[str(round)], safe=False)



def avg_grad(request):

    round = int(request.GET.get('round', -1))

    if round == -1:
        file_obj = File(JSON_PATH + 'avg_grad', 'avg_grad_')
        filename = file_obj.latest_file(ROUND_EVERY_FILE)
    else:
        filename = 'avg_grad_' + \
                   str((round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE) + '_' + \
                   str((round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE + ROUND_EVERY_FILE - 1) + '.json'

    file = open(JSON_PATH + 'avg_grad/' + filename, 'r', encoding='utf-8')
    data = json.load(file)
    file.close()

    if round == -1:
        return JsonResponse({'round': int(list(data.keys())[-1]), 'data': data[list(data.keys())[-1]]}, safe=False)
    else:
        return JsonResponse(data[str(round)], safe=False)


def trained_clients(request):

    round = int(request.GET.get('round', -1))
    num = int(request.GET.get('number', -1))


    if round == -1 or num == -1:
        return JsonResponse('Wrong Parameters', safe=False)


    files = []
    cur_round = round - num + 1
    while cur_round <= round:
        filename = 'gradients_' + \
               str((cur_round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE) + '_' + \
               str((cur_round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE + ROUND_EVERY_FILE - 1) + '.json'
        files.append(filename)
        cur_round = (cur_round // ROUND_EVERY_FILE) * ROUND_EVERY_FILE + ROUND_EVERY_FILE

    clients = {}
    for filename in files:
        file = open(JSON_PATH + 'client_grad/' + filename, 'r', encoding='utf-8')
        data = json.load(file)
        file.close()
        for key in data:
            if round - num + 1 <= int(key) <= round:
                clients[key] = list(map(int, list(data[key].keys())))

    return JsonResponse(clients, safe=False)

def weight(request):

    file = open(JSON_PATH + 'weight.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()

    return JsonResponse(data, safe=False)


def get_metrics_by_rounds(request):
    curRound = int(request.GET.get('round', -1))
    roundNum = int(request.GET.get('roundNum', -1))
    layer = rfile.get_layer(request.GET.get('layers', -1))
    res = []
    for round in range(curRound - roundNum + 1, curRound + 1):
        res.append(getOneRoundFromFile(round, layer))
    return JsonResponse(res, safe=False)

def one_round_metric(request):
    round = int(request.GET.get('round', -1))
    layer = rfile.get_layer(request.GET.get('layers', -1))
    res = getOneRoundFromFile(round, layer)
    return JsonResponse(res, safe=False)


with open('data/dense_metrics.pkl', 'rb') as fp:
    Dense_Metric = pkl.load(fp)


def get_all_round_metric(request):
    layers = rfile.get_layer(request.GET.getlist('layers[]', []))
    return JsonResponse({'res': Dense_Metric}, safe=False)

def get_multiple_information(request):
    start = int(request.GET.get('start', -1))
    end = int(request.GET.get('end', -1))
    layer = rfile.get_layer(request.GET.get('layers', -1))
    filter = request.GET.getlist('filter[]', [])
    multipleInfo = multiple_information(start, end, layer, filter)
    return JsonResponse({'res': multipleInfo}, safe=False)

def get_tsne_res(request):
    start = int(request.GET.get('start', -1))
    end = int(request.GET.get('end', -1))
    position = get_tsne(start, end)
    return JsonResponse({'res': position}, safe=False)

