from django.urls import path

from . import views

app_name = 'solve'

urlpatterns = [
    path('', views.solve, name='solve_template'),
]