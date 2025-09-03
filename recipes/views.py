from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

# Load model and processor once (for all requests)
MODEL_NAME = "prithivMLmods/Food-101-93M"
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model.eval()

# Food-101 labels
LABELS = {
    0: "apple_pie", 1: "baby_back_ribs", 2: "baklava", 3: "beef_carpaccio",
    4: "beef_tartare", 5: "beet_salad", 6: "beignets", 7: "bibimbap",
    8: "bread_pudding", 9: "breakfast_burrito", 10: "bruschetta",
    11: "caesar_salad", 12: "cannoli", 13: "caprese_salad", 14: "carrot_cake",
    15: "ceviche", 16: "cheesecake", 17: "cheese_plate", 18: "chicken_curry",
    19: "chicken_quesadilla", 20: "chicken_wings", 21: "chocolate_cake",
    22: "chocolate_mousse", 23: "churros", 24: "clam_chowder",
    25: "club_sandwich", 26: "crab_cakes", 27: "creme_brulee",
    28: "croque_madame", 29: "cup_cakes", 30: "deviled_eggs",
    31: "donuts", 32: "dumplings", 33: "edamame", 34: "eggs_benedict",
    35: "escargots", 36: "falafel", 37: "filet_mignon", 38: "fish_and_chips",
    39: "foie_gras", 40: "french_fries", 41: "french_onion_soup",
    42: "french_toast", 43: "fried_calamari", 44: "fried_rice",
    45: "frozen_yogurt", 46: "garlic_bread", 47: "gnocchi", 48: "greek_salad",
    49: "grilled_cheese_sandwich", 50: "grilled_salmon", 51: "guacamole",
    52: "gyoza", 53: "hamburger", 54: "hot_and_sour_soup", 55: "hot_dog",
    56: "huevos_rancheros", 57: "hummus", 58: "ice_cream", 59: "lasagna",
    60: "lobster_bisque", 61: "lobster_roll_sandwich", 62: "macaroni_and_cheese",
    63: "macarons", 64: "miso_soup", 65: "mussels", 66: "nachos",
    67: "omelette", 68: "onion_rings", 69: "oysters", 70: "pad_thai",
    71: "paella", 72: "pancakes", 73: "panna_cotta", 74: "peking_duck",
    75: "pho", 76: "pizza", 77: "pork_chop", 78: "poutine", 79: "prime_rib",
    80: "pulled_pork_sandwich", 81: "ramen", 82: "ravioli", 83: "red_velvet_cake",
    84: "risotto", 85: "samosa", 86: "sashimi", 87: "scallops",
    88: "seaweed_salad", 89: "shrimp_and_grits", 90: "spaghetti_bolognese",
    91: "spaghetti_carbonara", 92: "spring_rolls", 93: "steak",
    94: "strawberry_shortcake", 95: "sushi", 96: "tacos", 97: "takoyaki",
    98: "tiramisu", 99: "tuna_tartare", 100: "waffles",101: "chicken_biriyani",
    102: "payasam",
    103: "banana_fritters",
    104: "jackfruit_curry",
    105: "mango",
    106: "pineapple"
}

# Map dishes to main ingredients (can extend)
DISH_TO_INGREDIENTS = {
    "apple_pie": ["apple", "flour", "sugar", "butter"],
    "garlic_bread": ["bread", "garlic", "butter"],
    "spaghetti_bolognese": ["spaghetti", "tomato", "beef", "onion", "garlic"],
    "grilled_cheese_sandwich": ["bread", "cheese", "butter"],
    "samosa": ["potato", "peas", "flour", "spices"],
    "chicken_biriyani": ["chicken", "rice", "spices", "onion", "tomato"],
    "payasam": ["milk", "rice", "sugar", "cardamom", "cashew"],
    "banana_fritters": ["banana", "flour", "sugar", "oil"],
    "jackfruit_curry": ["jackfruit", "coconut", "spices", "onion"],
    "mango": ["mango"],
    "pineapple": ["pineapple"],
    "omelette": ["egg", "tomato", "onion"],
    "fried_rice": ["rice", "garlic", "onion"],
    "chicken_curry": ["chicken", "tomato", "onion", "garlic"],
    "pancakes": ["milk", "egg", "flour"],
    "pizza": ["flour", "tomato", "cheese", "capsicum"],
    "pad_thai": ["rice noodles", "egg", "tofu", "peanuts", "bean sprouts"],
    "risotto": ["rice", "cheese", "butter", "mushroom"],
    "paella": ["rice", "seafood", "chicken", "saffron"],
    "greek_salad": ["tomato", "cucumber", "onion", "feta", "olive"],
    "caprese_salad": ["tomato", "mozzarella", "basil"],
    "frozen_yogurt": ["milk", "sugar", "yogurt"],
    "ceviche": ["fish", "lime", "onion", "cilantro"],
    "cheese_plate": ["cheese", "crackers", "grapes"]  
}


class DetectIngredientsAPI(APIView):
    def post(self, request):
        if "image" not in request.FILES:
            return Response({"detail":"image is required"}, status=400)

        image_file = request.FILES["image"]
        image = Image.open(image_file).convert("RGB")

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

        # Get top 5 predicted dishes
        top_indices = torch.topk(probs, k=5).indices.tolist()
        predicted_dishes = [LABELS[idx] for idx in top_indices]
        print("Predicted dishes:", predicted_dishes)
        predicted_ingredients = set()
        for idx in top_indices:
            dish = LABELS.get(idx, "").lower().replace(" ", "_")  # normalize
            predicted_ingredients.update(DISH_TO_INGREDIENTS.get(dish, []))

        return Response({"ingredients": list(predicted_ingredients)})


class SuggestRecipesAPI(APIView):
    RECIPES = [
        {"id": 1, "title": "Tomato Omelette", "ingredients": ["egg","tomato","onion"], "steps": "Beat eggs, chop tomato & onion, cook on pan."},
        {"id": 2, "title": "Garlic Fried Rice", "ingredients": ["rice","garlic","onion"], "steps": "Cook rice, fry with garlic & onion."},
        {"id": 3, "title": "Chicken Curry", "ingredients": ["chicken","tomato","onion","garlic"], "steps": "Fry chicken with spices, add tomato & onion, simmer."},
        {"id": 4, "title": "Pancakes", "ingredients": ["milk","egg","flour"], "steps": "Mix ingredients, cook on griddle."},
    ]

    def post(self, request):
        user_ingredients = set(request.data.get("ingredients", []))
        result = []

        for r in self.RECIPES:
            r_ing = set(r["ingredients"])
            coverage = len(user_ingredients & r_ing) / len(r_ing) if r_ing else 0
            if coverage > 0:
                item = r.copy()
                item["coverage"] = round(coverage, 2)
                item["missing"] = list(r_ing - user_ingredients)
                result.append(item)

        # Sort by coverage descending
        result.sort(key=lambda x: x["coverage"], reverse=True)
        return Response({"results": result})


