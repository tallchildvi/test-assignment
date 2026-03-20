import argparse
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# BIO tagging scheme: B-ANI = beginning of animal, I-ANI = inside, O = other
LABEL2ID = {"O": 0, "B-ANI": 1, "I-ANI": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def generate_dataset(animals: list[str], num_samples: int = 1500) -> list[dict]:
    """
    Generate a scaled synthetic NER dataset using templates, animals, and random adjectives.
    Punctuation is pre-separated by spaces to ensure clean tokenization.
    """
    templates = [
        "There is a {adj} {animal} in the picture .",
        "I can see a {animal} here .",
        "This image shows a {adj} {animal} .",
        "Is this a {animal} ?",
        "That looks like a {adj} {animal} to me .",
        "The {animal} is clearly visible .",
        "Can you see the {adj} {animal} in this photo ?",
        "I think this is a {animal} .",
        "This must be a {adj} {animal} .",
        "Looks like a {animal} to me .",
        "Look at that {adj} {animal} !",
        "What a {adj} {animal} we have here ."
    ]
    
    adjectives = [
        "big", "small", "cute", "scary", "fluffy", "wild", 
        "beautiful", "tiny", "huge", "brown", "black", "white"
    ]

    data = []
    for _ in range(num_samples):
        animal = random.choice(animals)
        template = random.choice(templates)
        adj = random.choice(adjectives)

        # Format text and split by space (punctuation is already separated in templates)
        text = template.format(animal=animal, adj=adj)
        words = text.split()
        labels = []
        
        # Split multi-word animals like "polar bear" -> ["polar", "bear"]
        animal_parts = animal.split()

        i = 0
        while i < len(words):
            word_clean = words[i].lower()
            if word_clean == animal_parts[0].lower():
                # First word of animal gets B-ANI tag
                labels.append("B-ANI")
                # Remaining words of multi-word animal get I-ANI
                for j in range(1, len(animal_parts)):
                    if i + j < len(words):
                        labels.append("I-ANI")
                i += len(animal_parts)
            else:
                labels.append("O")
                i += 1

        data.append({"words": words, "labels": labels})
        
    return data

def tokenize_and_align(examples, tokenizer):
    """
    Tokenize words and align BIO labels to subword tokens.
    Subword tokens that are not the first piece of a word get label -100
    so they are ignored during loss computation.
    """
    tokenized = tokenizer(
        examples["words"],
        truncation=True,
        is_split_into_words=True,
        padding=True,
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # Special tokens [CLS], [SEP] — ignore
                aligned.append(-100)
            elif word_id != prev_word_id:
                # First subtoken of a word — keep original label
                aligned.append(LABEL2ID[labels[word_id]])
            else:
                # Subsequent subtokens — ignore to avoid double-counting
                aligned.append(-100)
            prev_word_id = word_id
        all_labels.append(aligned)

    tokenized["labels"] = all_labels
    return tokenized

def compute_metrics(eval_pred):
    """Compute token-level accuracy ignoring padding tokens (label == -100)."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true, pred = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:  # skip padding
                true.append(ID2LABEL[l])
                pred.append(ID2LABEL[p])

    correct = sum(t == p for t, p in zip(true, pred))
    accuracy = correct / len(true) if true else 0.0
    return {"accuracy": round(accuracy, 4)}

def train(args):
    # Animal classes specific to our dataset
    animals = [
        'dog', 'horse', 'elephant', 'butterfly', 
        'chicken','cat','cow', 'sheep', 
        'spider', 'squirrel'
    ]

    print(f"Generating synthetic NER dataset with {args.num_samples} samples...")
    raw_data = generate_dataset(animals, num_samples=args.num_samples)

    # Train-validation split
    train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)

    def to_hf(data):
        """Convert list of dicts to HuggingFace Dataset."""
        return Dataset.from_dict({
            "words": [d["words"] for d in data],
            "labels": [d["labels"] for d in data],
        })

    print("Loading tokenizer and model...")
    # Load pretrained tokenizer and model head for token classification
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # Replace pretrained classification head
    )

    tokenize_fn = lambda examples: tokenize_and_align(examples, tokenizer)

    # Tokenize and align labels for both splits
    train_dataset = to_hf(train_data).map(tokenize_fn, batched=True)
    val_dataset = to_hf(val_data).map(tokenize_fn, batched=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",            # Updated from evaluation_strategy to avoid warnings
        save_strategy="epoch",
        load_best_model_at_end=True,       # Restore best checkpoint after training
        metric_for_best_model="accuracy",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer, # <--- Тільки це змінити
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}...")
    trainer.train()

    # Save final best model and tokenizer together for easy inference loading
    best_path = f"{args.output_dir}/best_ner_model"
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"Model successfully saved to {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model for animal extraction")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name from HuggingFace")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=1500, help="Number of synthetic sentences to generate")
    parser.add_argument("--output_dir", type=str, default="./saved_models", help="Directory to save checkpoints and final model")

    args = parser.parse_args()
    train(args)

