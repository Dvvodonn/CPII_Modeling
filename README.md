# CPII_RealEstate

A custom-built real estate price prediction system developed for the Computer Programming II course. This project implements several machine learning models from scratch — including Linear Regression, Decision Trees, Random Forests, and Gradient Boosting — and evaluates their performance on a real-world housing dataset.

---

## Models Implemented

- **Linear Regression** – A baseline model assuming linear relationships.
- **Decision Trees** – Recursive partitioning based on feature thresholds.
- **Random Forests** – An ensemble of decision trees trained on bootstrap samples.
- **Gradient Boosting** – A sequential ensemble that optimizes for residual errors.

---

## Features

- Object-Oriented implementations of core regression models  
- Model training, hyperparameter tuning, and evaluation  
- Real estate dataset preprocessing pipeline  
- Utilities for comparing model performance  
- Modular, pip-installable project structure  
- Pytests for each model  

---

## 📁 Project Structure

```
CPII_RealEstate/
├── models/                 # Custom-built models (regression, tree-based)
├── training/               # Scripts to train and tune each model
├── utils/                  # Preprocessing and evaluation helpers
├── data/                   # Source housing dataset
├── outputs/                # Trained models and visualizations
├── notebooks/              # Jupyter experiments and EDA
├── tests/                  # Pytests
├── app.py                 # Streamlit UI for model inference
├── setup.py               # Package installation config
└── environment.yml        # Conda environment definition
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate real_estate_env
git clone https://github.com/Dvvodonn/CPII_RealEstate.git
cd CPII_RealEstate
pip install -e .
```

---

## Train a Model

```bash
python -m CPII_RealEstate.training.train_decision_trees --retrain
```

---

## Tune Hyperparameters

Input grid to be iterated over in the `tune_and_evaluate` method inside the training script.

```bash
python -m CPII_RealEstate.training.train_gradient_boosting --tune
```

---

## Testing

```bash
pytest
```

---

## Evaluation Outputs

- Trained models are saved in `/outputs` as `.pkl` files.  
- Performance metrics (MAE, RMSE, R²) are printed to console after training.

---

## Best Found Parameters

- **Linear Regression**:
    MAE: 145898.48

- **Decision Trees**:  
  Best params: `max_depth=10`, `min_sample_split=30` with MAE: 137571.77
  
  - **Random Forests**:  
  Best params: `n_estimators=200`, `max_depth=None`, `min_sample_split=20` with MAE: 128842.09

- **Gradient Boosting**:  
  Best params: `n_estimators=300`, `learning_rate=0.1`, `max_depth=5`, `min_sample_split=15` with MAE: 127542.15




---

## User Interface

After models are trained with selected parameters:

```bash
streamlit run CPII_RealEstate/app.py
```

Use the dropdown to select feature values.  
**View**, **Grade**, and **Condition** are 1–10 score variables.

---

## Known Issues

- Accuracy of optimized models dropped significantly and could not be recovered.  
- Some training scripts rely on hardcoded paths or manual parameter selection.

---

## Authors

Daveed Vodonenko, Darius Vulturu, Lucas Portela