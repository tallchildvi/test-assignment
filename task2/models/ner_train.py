import argparse
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# BIO tagging scheme
LABEL2ID = {"O": 0, "B-ANI": 1, "I-ANI": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def apply_typo(word):
    """Introduces a random character-level typo into a word."""
    if len(word) < 4: return word
    chars = list(word)
    idx = random.randint(0, len(chars) - 1)
    rnd = random.random()
    
    if rnd < 0.33: # Swap adjacent characters
        if idx < len(chars) - 1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif rnd < 0.66: # Remove a character
        chars.pop(idx)
    else: # Substitute with a random vowel
        chars[idx] = random.choice("aeiou")
    return "".join(chars)

def generate_dataset(animals: list[str], num_samples: int = 3000) -> list[dict]:
    """Generates synthetic NER dataset with typos and diverse templates."""
    templates = [
        "There is a {adj} {animal} in the picture .",
        "I can see a {animal} here .",
        "This image shows a {adj} {animal} .",
        "Is this a {animal} ?",
        "That looks like a {adj} {animal} to me .",
        "The {animal} is clearly visible .",
        "Can you see the {adj} {animal} in this photo ?",
        "I think this is a {animal} .",
        "Look at that {adj} {animal} !",
        # New templates where animal is at the start
        "{animal} is what I see in the frame .",
        "{animal} , that is the main subject .",
        "{animal} detected in the foreground .",
        "{animal} is the animal here .",
        "{animal} ! Look at it ."
    ]
    
    adjectives = ["big", "small", "cute", "scary", "fluffy", "wild", "tiny", "huge", "brown", "white"]
    data = []

    for _ in range(num_samples):
        base_animal = random.choice(animals)
        # Apply typo with 25% probability for robust training
        display_animal = apply_typo(base_animal) if random.random() < 0.25 else base_animal
        
        template = random.choice(templates)
        adj = random.choice(adjectives)
        
        # Build sentence
        text = template.format(animal=display_animal, adj=adj)
        words = text.split()
        
        # Labeling logic
        labels = []
        animal_parts = display_animal.split()
        
        i = 0
        while i < len(words):
            # Check if this word (or sequence) matches our current animal
            match = True
            for j in range(len(animal_parts)):
                if i + j >= len(words) or words[i+j].lower().strip(".,!?") != animal_parts[j].lower():
                    match = False
                    break
            
            if match:
                labels.append("B-ANI")
                for _ in range(1, len(animal_parts)):
                    labels.append("I-ANI")
                i += len(animal_parts)
            else:
                labels.append("O")
                i += 1

        data.append({"words": words, "labels": labels})
    return data

def tokenize_and_align(examples, tokenizer):
    """Aligns word-level BIO labels with subword tokens."""
    tokenized = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word_id:
                aligned.append(LABEL2ID[labels[word_id]])
            else:
                aligned.append(-100)
            prev_word_id = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized

def train(args):
    animals = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
    
    print(f"Generating augmented dataset with {args.num_samples} samples...")
    raw_data = generate_dataset(animals, num_samples=args.num_samples)
    train_data, val_data = train_test_split(raw_data, test_size=0.15, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True
    )

    def to_hf(data):
        return Dataset.from_dict({"words": [d["words"] for d in data], "labels": [d["labels"] for d in data]})

    train_ds = to_hf(train_data).map(lambda x: tokenize_and_align(x, tokenizer), batched=True)
    val_ds = to_hf(val_data).map(lambda x: tokenize_and_align(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: {"accuracy": np.mean(np.argmax(p.predictions, axis=-1)[p.label_ids != -100] == p.label_ids[p.label_ids != -100])}
    )

    print("Starting robust training...")
    trainer.train()
    
    out_path = Path(args.output_dir) / "best_ner_model"
    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"Success! Model saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default="./weights/ner")
    train(parser.parse_args())