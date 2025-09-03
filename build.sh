#!/bin/bash
# build.sh - Setup script for Django + Transformers project

# Exit on first error
set -e

echo "Starting build process..."

# 1. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 2. Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 3. Apply Django migrations
echo "Applying migrations..."
python manage.py migrate

# 4. Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# 5. Optional: Preload ML models (cache)
echo "Downloading Transformers model..."
python - <<END
from transformers import AutoImageProcessor, SiglipForImageClassification
model_name = "prithivMLmods/Food-101-93M"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
print("Model loaded successfully!")
END

echo "Build complete."
