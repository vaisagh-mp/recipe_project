from rest_framework import serializers
from .models import Recipe, Ingredient

class IngredientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ingredient
        fields = ['id', 'name']

class RecipeSerializer(serializers.ModelSerializer):
    ingredients = serializers.SerializerMethodField()

    class Meta:
        model = Recipe
        fields = ['id', 'title', 'steps', 'ingredients']

    def get_ingredients(self, obj):
        return [ri.ingredient.name for ri in obj.recipe_ingredients.all()]
