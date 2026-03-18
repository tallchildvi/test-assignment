from interface import MnistClassifierInterface
import numpy as np
class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm: str):
        self.models = {'cnn': ConvolutionalModel,
                       'rf': RandomForestModel,
                       'nn': FeedForwardModel}

        algorigthm = algorithm.lower()
        if algorigthm not in self.models:
            raise ValueError(f"Invalid values for algorithm: {algorithm}, expected values: {list(self.models)}")
        
        self._model = self.models[algorigthm]()
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the selected model
        """
        self._model.train(x_train, y_train)

    def predict(self, x_test):
        """
        Predicts labels using the selected model
        """
        return self._model.predict(x_test)