import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from task1.interface import MnistClassifierInterface
from utils.data_loader import get_loader
from sklearn.model_selection import train_test_split
from pathlib import Path

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers to extract features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers (64 channels * 7 * 7 image size after pooling)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Prevents overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Apply convolution, activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten (batch_size, 64, 7, 7) to (batch_size, 3136)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class ConvolutionalModel(MnistClassifierInterface):
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = CNN().to(self.device)
        # Loss function for classification 
        self.criterion = nn.CrossEntropyLoss()
        # Adaptive optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # History of epochs
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, x_train, y_train, augmentation=False):
        # Split data on train and validation parts
        x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        
        train_loader = get_loader(x_t, y_t, batch_size=64, augmentation=augmentation)
        val_loader = get_loader(x_v, y_v, batch_size=64, augmentation=False)
        # Enable training mode
        self.model.train()
        epochs = 20 
        # Train loop
        for epoch in range(epochs):
            total_train_loss = 0.0
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
                total_train_loss += loss.item()

            # Switching to model testing
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    val_loss = self.criterion(self.model(images), labels)
                    total_val_loss += val_loss.item()
            
            # Save history for plotting
            self.history['train_loss'].append(total_train_loss / len(train_loader))
            self.history['val_loss'].append(total_val_loss / len(val_loader))
            print(f"Epoch {epoch+1}: Loss {self.history['train_loss'][-1]:.4f} | Val Loss {self.history['val_loss'][-1]:.4f}")
            # Switch to training mode
            self.model.train() 

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