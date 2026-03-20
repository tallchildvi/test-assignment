from torch.utils.data import DataLoader, Dataset


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

