import argparse
import torch
from transformers import pipeline

class AnimalNER:
    """Class for extracting animal entities from text using a fine-tuned Transformer model."""
    
    def __init__(self, model_path: str):
        # Initialize the NER pipeline with automatic entity grouping (merges B-ANI and I-ANI)
        self.ner_pipeline = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )

    def predict(self, text: str) -> list[str]:
        """Extracts unique lowercase animal names from the input string."""
        if not text.strip():
            return []

        # Run inference: returns list of dicts with 'word', 'entity_group', etc.
        results = self.ner_pipeline(text)
        
        # Filter entities containing "ANI" tags and normalize strings
        animals = []
        for entity in results:
            if "ANI" in entity['entity_group']:
                name = entity['word'].strip().lower()
                if name not in animals:
                    animals.append(name)
        
        return animals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal Extraction Inference")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--model", type=str, default="./weights/ner/best_ner_model", help="Model path")
    args = parser.parse_args()

    # Execute prediction and display results
    try:
        ner = AnimalNER(args.model)
        found = ner.predict(args.text)
        print(f"Text: {args.text}")
        print(f"Detected: {found}")
    except Exception as e:
        print(f"Inference Error: {e}")