from django.http import HttpResponse
from django.http import JsonResponse
import json

PATH = '/home/zty_11621014/federated/models/gradients/'


# def hello(request):
#  return HttpResponse("Hello world!")



def test(request):
    file = open(PATH + 'gradients.json', 'r', encoding='utf-8')
    data = json.load(file)
    return data


