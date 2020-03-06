from django.http import HttpResponse
from django.http import JsonResponse
import json


from backend.settings import JSON_PATH

# PATH = '/home/zty_11621014/federated/models/gradients/'


# def hello(request):
#  return HttpResponse("Hello world!")



def performance(request):
    file = open(JSON_PATH + 'performance.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    return JsonResponse(data,safe=False)

def client_grad(request):
    file = open(JSON_PATH + 'gradients.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    return JsonResponse(data, safe=False)

def avg_grad(request):
    file = open(JSON_PATH + 'avg_grad.json', 'r', encoding='utf-8')
    data = json.load(file)
    file.close()
    return JsonResponse(data, safe=False)




