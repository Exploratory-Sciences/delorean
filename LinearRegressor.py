import numpy as np

class LinearRegressor:
    """Linear Regression class, built for NumPy arrays. 
    Fits using OLS. Always supplies a constant coefficient.
    Very unstable!"""
    
    def __init__(self):
        self.X = None
        self.y = None
        self.B = None
        self.y_hat = None
        self._estimator_type = "regressor"
        
    
    def fit(self, X, y):
        for_coeff = np.ones((X.shape[0], 1))
        X = np.hstack((X, for_coeff))
        self.B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.X = X
        self.y = y
    
    
    def predict(self, X):
        if self.B is None:
            return None
        self.y_hat = self.X.dot(self.B)
        return self.y_hat

    
    def r_sqrd(self):
        if self.y is None:
            return None
        elif self.y_hat is None:
            y_hat = self.predict()
            
        y_bar = np.mean(self.y)
        explained = np.sum(np.power(self.y_hat - y_bar, 2))
        total = np.sum(np.power(self.y - y_bar, 2))
        return explained / total