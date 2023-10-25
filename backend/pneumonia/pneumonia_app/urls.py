from django.urls import path
from . import views

urlpatterns = [
    path('', views.my_view, name='hello_world'),
    path('predict/', views.predict, name='predict'),
]
