from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.start, name='start'),
    url(r'^search$', views.search, name='search'),
    url(r'^analyse$', views.analyse, name='analyse'),
]
