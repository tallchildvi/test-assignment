from task1.interface import MnistClassifierInterface
from task1.models.rf_model import RandomForestModel
from pathlib import Path
import numpy as np
class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm: str):
        # self.models = {'cnn': ConvolutionalModel,
        #                'rf': RandomForestModel,
        #                'nn': FeedForwardModel}
        self.models = {'rf': RandomForestModel}
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
    
    def save(self, path):
        """
        Save model.
        """
        save_path = Path(path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        self._model.save(path)

    def load(self, path):
        """
        Load model.
        """
        self._model.load()