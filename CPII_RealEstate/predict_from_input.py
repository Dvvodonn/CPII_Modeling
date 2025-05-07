import pickle 
import numpy as np

#ft names in order on which model was trained on
feature_names = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','view','condition','grade','yr_built','yr_renovated']

def load_model(path="outputs/tree_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def collect_user_input():
    inputs = []
    for feature in feature_names:
        while True:
            try:
                val = float(input(f'{feature}:?'))
                inputs.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    return inputs

def predict_from_features(model, features):
    X_input = np.array([features])
    prediction = model.predict(X_input)
    return prediction[0]

def main():
    model = load_model()
    features = collect_user_input()
    price = predict_from_features(model, features)
    print(f"\nPredicted house price: ${price:,.2f}")

if __name__ == "__main__":
    main()