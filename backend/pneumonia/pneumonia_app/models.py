from django.db import models

# Create your models here.
# models.py (inside your app)

import joblib

def load_ml_model(model_path):
    return joblib.load(model_path)
