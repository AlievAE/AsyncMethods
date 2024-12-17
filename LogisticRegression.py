import numpy as np

def sigmoid(z):
    """
    Compute sigmoid function
    
    Args:
        z (np.ndarray): Input values
    
    Returns:
        np.ndarray: Sigmoid of input values
    """
    return 1 / (1 + np.exp(-np.clip(z, -100, 100)))  # clip to avoid overflow

def logistic_regression_loss(data, x, lambda_reg=0.1):
    """
    Binary Cross-Entropy loss function for logistic regression with L2 regularization
    
    Args:
        data (tuple): Tuple containing (X, y) where:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,) with values in {0, 1}
        x (np.ndarray): Weight vector of shape (n_features,)
        lambda_reg (float): Regularization strength (default: 0.1)
    
    Returns:
        float: Cross-entropy loss value + L2 regularization term
    """
    X, y = data
    z = X @ x
    predictions = sigmoid(z)
    
    # Compute binary cross-entropy loss
    eps = 1e-15  # small constant to avoid log(0)
    predictions = np.clip(predictions, eps, 1 - eps)
    cross_entropy = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    
    # Add L2 regularization term
    reg_term = lambda_reg * np.sum(x ** 2)
    return cross_entropy + reg_term

def logistic_regression_gradient(data, x, batch_size=None, lambda_reg=0.1):
    """
    Compute stochastic gradient for logistic regression with cross-entropy loss and L2 regularization
    
    Args:
        data (tuple): Tuple containing (X, y) where:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,) with values in {0, 1}
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
    
    z = X @ x
    predictions = sigmoid(z)
    grad = X.T @ (predictions - y) / len(y)
    
    reg_grad = 2 * lambda_reg * x
    return grad + reg_grad
