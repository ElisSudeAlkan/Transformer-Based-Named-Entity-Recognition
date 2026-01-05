# pipeline.py
import json, argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input test file (json)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the prediction results (json)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_load_path}...")
    ner = pipeline(
        "token-classification",
        model=args.model_load_path,
        tokenizer=args.model_load_path,
        aggregation_strategy="simple"
    )

    print(f"Reading input from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    print("Running predictions...")
    
    # Process in batches or one by one. One by one is fine for this scale.
    total = len(data)
    for i, item in enumerate(data):
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{total}")
            
        text = " ".join(item["tokens"])
        
        # Handle long sequences if necessary, but pipeline usually handles it or truncates.
        # Ideally we should truncate if too long, or the pipeline config handles it.
        # For simplicity, we assume standard behavior.
        try:
            preds = ner(text)
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            preds = []

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

    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("✅ Predictions written")

if __name__ == "__main__":
    main()
