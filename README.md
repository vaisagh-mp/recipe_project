# Food Ingredient Detection & Recipe Suggestion API

This project is a Django REST Framework API that allows users to:  
1. Upload an image of food and detect its ingredients using a fine-tuned Food-101 image classification model.  
2. Suggest recipes based on the detected ingredients.

---

## Features

- Image-based food ingredient detection using **Hugging Face Transformers** (`SiglipForImageClassification` model).  
- Suggests recipes based on available ingredients with coverage and missing ingredients information.  
- Supports custom dishes, Kerala foods, fruits, and desserts.  
- Easy-to-extend `DISH_TO_INGREDIENTS` mapping.  

---

## Installation

1. Download the project ZIP file from the repository or source.  
2. Extract the ZIP file to a folder of your choice.  
3. Navigate to the extracted project folder:

```bash
cd <extracted-folder-name>

pip install -r requirements.txt

