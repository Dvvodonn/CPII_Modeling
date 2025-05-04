import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None  # will include bias as first term

    def _add_bias(self, X):
        """
        Adds a bias column (of 1s) to the feature matrix.
        This allows us to learn the intercept/bias term as part of the weights.
        """
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        """
        Fit linear regression model using the Normal Equation:
        β = (X^TX)⁻¹ X^Ty
        """
        X_b = self._add_bias(X)  # Add bias term
        XT_X = X_b.T @ X_b       # X^TX
        XT_y = X_b.T @ y         # X^Ty
        self.coefficients = np.linalg.inv(XT_X) @ XT_y  # β = (X^TX)^(-1) X^Ty

    def predict(self, X):
        """
        Predict values using the trained model.
        """
        if self.coefficients is None:
            raise ValueError("Model has not been trained yet.")
        X_b = self._add_bias(X)
        return X_b @ self.coefficients

    def score(self, X, y):
        """
        Calculate the R² score to evaluate model performance.
        R² = 1 - (SS_res / SS_tot) = 1 - (model's total error - total variance of the data)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
