from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.home, name='home'),
    path('result', views.result, name='result'), # type: ignore
    path('report', views.report, name='report'),
    path('statistics', views.statistics, name='statistics')
]
