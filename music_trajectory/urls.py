from django.contrib import admin
from django.urls import path
from predictor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.dashboard, name='dashboard'),
    path('dataset/', views.dataset, name='dataset'),
    path('predict/', views.predict_view, name='predict'),
    path('model/', views.model_report, name='model_report'),
    path('train/', views.train_model, name='train_model'),
    path('train-template/', views.train_template, name='train_template'),
    path('songs/<str:song_id>/', views.song_detail, name='song_detail'),
    path('api/live/', views.live_api, name='live_api'),
    path('api/dashboard/', views.dashboard_api, name='dashboard_api'),
]
