import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from task1.interface import MnistClassifierInterface
from utils.data_loader import get_loader
from pathlib import Path

class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        
        # transform (1, 28, 28) to (784)
        self.flatten = nn.Flatten()
        
        # layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Prevents overfitting
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeedForwardModel(MnistClassifierInterface):
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = FFNN().to(self.device)
        # Loss function for classification 
        self.criterion = nn.CrossEntropyLoss()
        # Adaptive optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, x_train, y_train, augmentation=False):

        train_loader = get_loader(x_train, y_train, batch_size=64, augmentation=augmentation)
        # Enable training mode
        self.model.train()
        epochs = 20 
        # Trian loop
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                # Reset gradients
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # Backpropogation
                loss.backward()
                # Update weights
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    def predict(self, x_test):
        # Disable dropout for inference
        self.model.eval()

        # Ensure batch dimentions
        if len(x_test.shape) == 2:
            x_test = x_test[None, None, ...]
        elif len(x_test.shape) == 3:
            x_test = x_test[:, None, ...]

        test_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)

        # Disable gradient tracking
        with torch.no_grad():
            outputs = self.model(test_tensor)
            # Get highest logit index
            _, predicted = torch.max(outputs, 1)
            
        return predicted.cpu().numpy()
    
    def save(self, path):
        """Save model weights to a file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Save state_dict (weights only)
        torch.save(self.model.state_dict(), path)
        print(f"Weights saved to {path}")

    def load(self, path):
        """Load weights from a file and set model to eval mode."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval() 
        print(f"Weights loaded from {path}")