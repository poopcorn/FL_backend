from django.http import HttpResponse
from django.http import JsonResponse


def hello(request):
 return HttpResponse("Hello world!")



def test(request):
    test_data = [{
        'name': 'aaa',
        'age': 12
    },{
        'name': 'bbb',
        'age': 15
    }]

    return JsonResponse({'ret': 0, 'relist': test_data})