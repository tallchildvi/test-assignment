from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
from utils.dataset_distribution import calculate_class_weights
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

class AnimalDataset(Dataset):
    """Custom PyTorch Dataset for Hugging Face datasets."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Convert to RGB to handle any grayscale or RGBA images
        image = item['image'].convert('RGB')
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    # 1. Load Data
    full_dataset = load_dataset("imagefolder", data_dir="./data", split="train")

    # Calculate weights based on the entire distribution before splitting
    class_weights = calculate_class_weights(full_dataset).to(device)
    print(f"Calculated class weights: {class_weights}")
    
    # Split into train (85%) and validation (15%)
    splited_data = full_dataset.train_test_split(test_size=0.15, seed=42)
    
    # Define Transforms
    train_transform = transforms.Compose([
        # ResNet expects 224x224
        transforms.Resize((224, 224)),

        # Random augmentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),

        transforms.ToTensor(),

        # Normalizing each color (RGB) using ImageNet statistics (mean and standard deviation)
        # For transfer learning to work correctly 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Same as train data transformation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #  Create DataLoaders
    train_loader = DataLoader(
        AnimalDataset(splited_data['train'], transform=train_transform), 
        batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        AnimalDataset(splited_data['test'], transform=val_transform), 
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Initialize Model (Transfer Learning)
    print("Initializing pre-trained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the final layer to match 10 classse
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #  Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Switching to model testing
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "best_cv_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")

    print("Train complete!")

# Make this file module
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="./saved_models")
    
    args = parser.parse_args()
    train(args)