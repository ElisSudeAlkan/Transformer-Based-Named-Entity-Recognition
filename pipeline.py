# pipeline.py
import json, argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/ner_model")
    parser.add_argument("--input_file", type=str, default="data/dataset1.json")
    parser.add_argument("--output_file", type=str, default="outputs/predictions.json")
    args = parser.parse_args()

    ner = pipeline(
        "token-classification",
        model=args.model_path,
        tokenizer=args.model_path,
        aggregation_strategy="simple"
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        text = " ".join(item["tokens"])
        preds = ner(text)

        clean_preds = []
        for p in preds:
            clean_preds.append({
                "entity_group": p["entity_group"],
                "score": float(p["score"]),
                "word": p.get("word", ""),
                "start": int(p.get("start", 0)),
                "end": int(p.get("end", 0))
            })

        results.append({
            "text": text,
            "predictions": clean_preds
        })

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("✅ Predictions written")

if __name__ == "__main__":
    main()
