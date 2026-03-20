import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageClassificationInterface:
    """Interface for Image Classification Models"""
    def predict(self, image_path: str) -> str:
        raise NotImplementedError

class AnimalClassifier(ImageClassificationInterface):
    """
    Implementation of the Image Classification model using ResNet18.
    """
    def __init__(self, model_path: str, num_classes: int = 10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ResNet18 and modify the final layer using custom weights
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Successfully loaded CV model weights from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model weights not found at {model_path}. Please train the model first.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Standard transformations for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Map class ids to labels
        self.id_to_class = {
            0: 'dog', 1: 'horse', 2: 'elephant', 3: 'butterfly', 
            4: 'chicken', 5: 'cat', 6: 'cow', 7: 'sheep', 
            8: 'spider', 9: 'squirrel'
        }

    def predict(self, image_path: str):
        """
        Takes an image path, processes it, and returns the predicted animal class.
        """
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            
        return self.id_to_class[predicted_idx.item()]

# Make this file module
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test CV Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="./saved_models/best_cv_model.pth", help="Path to model weights")
    args = parser.parse_args()
    
    classifier = AnimalClassifier(model_path=args.model)
    prediction = classifier.predict(args.image)
    print(f"Predicted class: {prediction}")