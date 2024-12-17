import numpy as np

def linear_regression_loss(data, x, lambda_reg=0.1):
    """
    Mean Squared Error loss function for linear regression with L2 regularization
    
    Args:
        data (tuple): Tuple containing (X, y) where:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
        x (np.ndarray): Weight vector of shape (n_features,)
        lambda_reg (float): Regularization strength (default: 0.1)
    
    Returns:
        float: MSE loss value + L2 regularization term
    """
    X, y = data
    predictions = X @ x
    mse = np.mean((predictions - y) ** 2)
    reg_term = lambda_reg * np.sum(x ** 2)
    return mse + reg_term

def linear_regression_gradient(data, x, batch_size=None, lambda_reg=0.1):
    """
    Compute stochastic gradient for linear regression with MSE loss and L2 regularization
    
    Args:
        data (tuple): Tuple containing (X, y) where:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
        x (np.ndarray): Weight vector of shape (n_features,)
        batch_size (int, optional): Size of random batch to use. If None, use full dataset
        lambda_reg (float): Regularization strength (default: 0.1)
    
    Returns:
        np.ndarray: Gradient vector of shape (n_features,)
    """
    X, y = data
    
    if batch_size is not None:
        indices = np.random.choice(len(y), size=min(batch_size, len(y)), replace=False)
        X = X[indices]
        y = y[indices]
    
    predictions = X @ x
    grad = 2 * X.T @ (predictions - y) / len(y)
    reg_grad = 2 * lambda_reg * x
    return grad + reg_grad