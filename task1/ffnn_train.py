from utils.data_loader import get_mnist_dataset
from task1.classifier import MnistClassifier
import numpy as np

# Receive data
X_train, X_test, y_train, y_test = get_mnist_dataset()
# Initialize model
clf = MnistClassifier(algorithm='ffnn')
# Train model
clf.train(X_train, y_train, augmentation=True)
# Save model
clf.save("weights/ffnn_model.pth")
