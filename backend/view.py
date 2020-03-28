from django.http import HttpResponse
from django.http import JsonResponse
import json


from backend.settings import JSON_PATH
from backend.settings import ROUND_EVERY_FILE
from backend.file import File



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





