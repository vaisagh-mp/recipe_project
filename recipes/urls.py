from django.urls import path
from .views import DetectIngredientsAPI, SuggestRecipesAPI

urlpatterns = [
    path('ingredients/detect/', DetectIngredientsAPI.as_view()),
    path('recipes/suggest/', SuggestRecipesAPI.as_view()),
]


