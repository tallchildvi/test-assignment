from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def get_mnist_dataset(test_size: float = 0.3, random_state: int = 42):
    """
    Load, scale, and split MNIST dataset.

    Args:
        test_size: Fraction of the dataset for testing.
        random_state: Seed for reproducible split.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
    
    # Scale pixels to [0, 1] and cast to float32
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int64')
    
    # Reshape to (samples, height, width)
    X = X.reshape(-1, 28, 28)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)