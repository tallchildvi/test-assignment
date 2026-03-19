from abc import ABC, abstractmethod
import numpy as np

class MnistClassifierInterface(ABC):
    """
    Abstract class for MNIST classification models
    """
    @abstractmethod
    def train(self, x_train:np.ndarray, y_train: np.ndarray):
        """
        Train the model using features and labels.

        Args:
            x_train (np.ndarray): Training data;
            y_train (np.ndarray): Training labels.

        """
        pass

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict labels from the given data.

        Args: 
            x_test (np.ndarray) : Input data to classify. 

        Returns:
            np.ndarray: Predicted labels.
        """
        pass
    @abstractmethod
    def save(self, path: str):
        """
        Save model to the given path.

        Args:
            path (str): Path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load model from the given path.

        Args: 
            path (str): Path to load the model
        """
        pass