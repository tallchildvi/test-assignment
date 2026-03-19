from utils.data_loader import get_mnist_dataset
from task1.classifier import MnistClassifier

# initialize model
clf = MnistClassifier(algorithm='rf')
# receive data
X_train, X_test, y_train, y_test = get_mnist_dataset()
# train model
clf.train(x_train=X_train, y_train=y_train)
# save model
clf.save("weights/rf_model.joblib")
print("Random Forest model saved to weights/rf_model.joblib")
#test model
preds = clf.predict(x_test=X_test)
accuracy = (preds == y_test).mean()
print(f"Accuracy: {accuracy*100}%")