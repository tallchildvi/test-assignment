from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def get_mnist_dataset(test_size: float = 0.2, random_state: int = 42):
    """
    Load, scale, and split MNIST dataset.

    Args:
        test_size: Fraction of the dataset for testing.
        random_state: Seed for reproducible split.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """ 
    mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
    
    # Scale pixels to [0, 1] and cast to float32
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int64')
    
    # Reshape to (samples, height, width)
    X = X.reshape(-1, 28, 28)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

class MnistDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        image = (self.images[idx] * 255).astype('uint8')
        image = Image.fromarray(image)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def get_loader(x, y, batch_size=64, augmentation=False):
    """
    Standard loader. 
    Set augmentation=True for training  only.
    """
    t_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    # Add image transfomation
    if augmentation:
        t_list = [
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, (0.1, 0.1))
        ] + t_list

    dataset = MnistDataset(x, y, transform=transforms.Compose(t_list))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)