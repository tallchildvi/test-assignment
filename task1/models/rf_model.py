import joblib
from sklearn.ensemble import RandomForestClassifier
from task1.interface import MnistClassifierInterface
import numpy as np
class RandomForestModel(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
    def train(self, x_train, y_train):
        #reshape images to 1 dim array
        n_samples = x_train.shape[0]
        x_flattened = x_train.reshape(n_samples, -1)

        #train random forest
        self.model.fit(x_flattened, y_train)
    
    def predict(self, x_test):
        #reshape image to 1 dim array
        n_samples = x_test.shape[0]
        x_flattened = x_test.reshape(n_samples, -1)

        #make prediction with random forest
        return self.model.predict(x_flattened)
    
    def save(self, path):
        joblib.dump(self.model, filename=path, compress=3)

    def load(self, path):
        self.model = joblib.load(path)