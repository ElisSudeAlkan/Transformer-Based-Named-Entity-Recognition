# train.py
import json, argparse, os, random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Directory containing dataset1.json and dataset2.json")
    parser.add_argument("--model_save_path", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--num_train_epoch", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    args = parser.parse_args()

    # Load datasets
    ds1_path = os.path.join(args.dataset_path, "dataset1.json")
    ds2_path = os.path.join(args.dataset_path, "dataset2.json")

    print(f"Loading datasets from {args.dataset_path}...")
    dataset1 = load_json(ds1_path)
    dataset2 = load_json(ds2_path)

    # Split dataset1 (85% train, 15% test)
    # Ensure reproducibility
    random.seed(42)
    random.shuffle(dataset1)
    
    split_idx = int(len(dataset1) * 0.85)
    ds1_train = dataset1[:split_idx]
    ds1_test = dataset1[split_idx:]

    print(f"Dataset1 split: {len(ds1_train)} train, {len(ds1_test)} test")
    
    # Save the test split for pipeline.py usage
    test_file_path = os.path.join(args.dataset_path, "dataset1_test_split.json")
    save_json(ds1_test, test_file_path)
    print(f"Saved 15% test split to {test_file_path}")

    # Combine dataset2 with train split of dataset1
    full_train_data = ds1_train + dataset2
    print(f"Total training examples (Dataset1_train + Dataset2): {len(full_train_data)}")

    # Filter out malformed data
    original_len = len(full_train_data)
    full_train_data = [x for x in full_train_data if len(x["tokens"]) == len(x["labels"])]
    if len(full_train_data) < original_len:
        print(f"Removed {original_len - len(full_train_data)} malformed examples (token/label length mismatch)")

    # Extract labels
    label_set = sorted({label for item in full_train_data for label in item["labels"]})
    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}

    # Convert to HF Dataset
    def to_hf(example):
        return {
            "tokens": example["tokens"],
            "labels": [label2id[l] for l in example["labels"]]
        }

    hf_train = Dataset.from_list([to_hf(x) for x in full_train_data])
    hf_test = Dataset.from_list([to_hf(x) for x in ds1_test])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing data...")
    tokenized_train = hf_train.map(tokenize_and_align, batched=True)
    tokenized_test = hf_test.map(tokenize_and_align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_set),
        label2id=label2id,
        id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epoch,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.model_save_path}")
    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)

    # Save label mapping for pipeline usage
    # Since we can just load it from config.json in the model dir, this isn't strictly necessary 
    # but good for inspection.
    
    print("✅ Training finished")

if __name__ == "__main__":
    main()
