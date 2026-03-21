import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import pipeline
import argparse

class AnimalVerificationPipeline:

    def __init__(self, ner_path: str, cv_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize NER model (Transformer-based)
        self.ner_pipeline = pipeline(
            "ner",
            model=ner_path,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 2. Initialize Image Classification model (ResNet18)
        self.cv_model = models.resnet18()
        self.cv_model.fc = nn.Linear(self.cv_model.fc.in_features, 10)
        self.cv_model.load_state_dict(torch.load(cv_path, map_location=self.device))
        self.cv_model.to(self.device)
        self.cv_model.eval()
        
        # Mapping Italian dataset folder names to English classes for comparison
        self.idx_to_animal = {
            0: 'dog',       # cane
            1: 'horse',     # cavallo
            2: 'elephant',  # elefante
            3: 'butterfly', # farfalla
            4: 'chicken',   # gallina
            5: 'cat',       # gatto
            6: 'cow',       # mucca
            7: 'sheep',     # pecora
            8: 'spider',    # ragno
            9: 'squirrel'   # scoiattolo
        }
        
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def run(self, text: str, image_path: str) -> bool:
        """
        Main inference logic. Returns True if text and image match, else False.
        """
        # Step 1: Extract animal names from text
        ner_results = self.ner_pipeline(text)
        text_animals = [ent['word'].lower().strip() for ent in ner_results if "ANI" in ent['entity_group']]
        
        # Step 2: Classify animal in the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.img_transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.cv_model(img_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            image_animal = self.idx_to_animal[pred_idx]
        
        # Step 3: Boolean verification
        # Returns True if the animal detected in the image is mentioned in the text
        return image_animal in text_animals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Animal Verification Pipeline")
    parser.add_argument("--text", type=str, required=True, help="Input text description")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--ner_path", type=str, default="./weights/ner/best_ner_model")
    parser.add_argument("--cv_path", type=str, default="./weights/cv/best_cv_model.pth")
    
    args = parser.parse_args()
    
    # Initialize and execute
    try:
        pipeline_system = AnimalVerificationPipeline(args.ner_path, args.cv_path)
        result = pipeline_system.run(args.text, args.image)
        
        # Output strictly according to task requirements
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
