import numpy as np
import torch

def calculate_class_weights(dataset):
    """
    Calculates class weights to handle dataset imbalance.
    Returns a tensor of weights for CrossEntropyLoss.
    """
    labels = dataset['label']
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Weight formula: total_samples / (num_classes * count_for_class)
    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)