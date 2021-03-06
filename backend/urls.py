"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import view

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('performance/', view.performance),
    path('client_grad/', view.client_grad),
    path('avg_grad/', view.avg_grad),
    path('trained_clients/', view.trained_clients),
    path('weight/', view.weight),
    path('anomaly/', include('anomaly.urls')),
    path('contribution/', include('contribution.urls')),
    path('one_round_metrics/', view.one_round_metric),
    path('one_round_all_grad', view.all_grad),
    # path('all_round_metrics/', view.get_all_round_metric)
]
