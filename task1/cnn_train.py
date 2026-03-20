from utils.data_loader import get_mnist_dataset
from task1.classifier import MnistClassifier
import matplotlib.pyplot as plt
import numpy as np

# Receive data
X_train, X_test, y_train, y_test = get_mnist_dataset()
# Initialize model
clf = MnistClassifier(algorithm='cnn')
# Train model
clf.train(X_train, y_train, augmentation=True)
# Save model
clf.save("weights/cnn_model.pth")

history = clf._model.history

plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
print(history)
plt.show()