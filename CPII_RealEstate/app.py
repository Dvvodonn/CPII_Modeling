import os
import sys
from pathlib import Path
import pickle
import streamlit as st
import numpy as np

# Ensure project root is on the path so your package imports work
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Custom unpickler to remap the 'outputs.random_forest_model' module path

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect old module path to your actual model code location
        if module == 'CPII_RealEstate.outputs.random_forest_model':
            module = 'CPII_RealEstate.models.random_forest'
        return super().find_class(module, name)

# Import FEATURES from your CLI helper
from CPII_RealEstate.predict_from_input import FEATURES

# Helper for sensible defaults
def sensible_default(name):
    return 1 if name in ("bedrooms", "bathrooms") else 0

# Locate outputs folder inside CPII_RealEstate
o = THIS_DIR / "outputs"
model_files = list(o.glob("*.pkl"))
if not model_files:
    st.error(f"No model files found in {o}/")
    st.stop()

# Dropdown for model selection
chosen = st.selectbox("Select model", [m.name for m in model_files])
model_path = o / chosen

# Load the selected model with remapping
with open(model_path, "rb") as f:
    model = RenameUnpickler(f).load()

st.title("🏠 House Price Predictor")
st.write(f"**Using model:** `{chosen}`")

# Build input form
inputs = {}
for name, dtype in FEATURES:
    default = sensible_default(name)
    fmt = "%d" if dtype is int else "%f"
    inputs[name] = st.number_input(name, value=default, format=fmt)

# Predict and display
if st.button("Predict"):
    x = __import__('numpy').array([[inputs[f] for f,_ in FEATURES]])
    price = model.predict(x)[0]
    st.metric("Predicted Price", f"${price:,.2f}")
