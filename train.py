# train.py
import json, argparse, os
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--output_dir", type=str, default="outputs/ner_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    dataset1 = load_json(os.path.join(args.data_dir, "dataset1.json"))
    dataset2 = load_json(os.path.join(args.data_dir, "dataset2.json"))

    # 🔹 merge datasets
    full_data = dataset1 + dataset2

    # 🔹 extract labels
    label_set = sorted({label for item in full_data for label in item["labels"]})
    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}

    # 🔹 convert to HF Dataset
    def to_hf(example):
        return {
            "tokens": example["tokens"],
            "labels": [label2id[l] for l in example["labels"]]
        }

    hf_dataset = Dataset.from_list([to_hf(x) for x in full_data])
    hf_dataset = hf_dataset.train_test_split(test_size=0.15, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True
        )
        tokenized["labels"] = examples["labels"]
        return tokenized

    tokenized_ds = hf_dataset.map(tokenize_and_align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_set),
        label2id=label2id,
        id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_list.json"), "w") as f:
        json.dump(label_set, f, indent=2)

    print("✅ Training finished")

if __name__ == "__main__":
    main()
