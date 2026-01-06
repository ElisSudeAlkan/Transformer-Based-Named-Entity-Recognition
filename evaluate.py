import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_overlap(span1, span2):
    """
    Returns true if span1 overlaps with span2
    span: (start, end)
    """
    return max(0, min(span1[1], span2[1]) - max(span1[0], span2[0])) > 0

def matching_score(token_span, entity_span):
    # simple overlap check
    # ensure reasonable overlap? token is fully inside entity or vice versa
    # For now, just any overlap
    start = max(token_span[0], entity_span[0])
    end = min(token_span[1], entity_span[1])
    overlap = max(0, end - start)
    return overlap

def evaluate():
    try:
        with open('data/dataset1_test_split.json', 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        with open('outputs/predictions.json', 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    if len(ground_truth) != len(predictions):
        print(f"Warning: Number of instances mismatch. GT: {len(ground_truth)}, Pred: {len(predictions)}")
    
    total_tokens = 0
    correct_tokens = 0
    
    # Store lists for generic classification report if needed
    y_true_all = []
    y_pred_all = []

    for i, gt_item in enumerate(ground_truth):
        if i >= len(predictions):
            break
            
        pred_item = predictions[i]
        
        gt_tokens = gt_item['tokens']
        # Depending on how the pipeline processed text, we try to reconstruct offsets
        # The prediction 'text' field should be used as reference
        text = pred_item['text']
        
        # We need to find token offsets in 'text'. 
        # A simple strategy is to find tokens sequentially.
        
        token_spans = []
        current_pos = 0
        
        for token in gt_tokens:
            # Find token in text starting from current_pos
            # Use simple find, assuming tokens appear in order
            start_idx = text.find(token, current_pos)
            if start_idx == -1:
                # Fallback or error if token not found (e.g. normalization issues)
                # For robustness, just advance strictly
                # This might happen if spaces are different.
                start_idx = current_pos # naive fallback
                pass
            
            end_idx = start_idx + len(token)
            token_spans.append((start_idx, end_idx))
            current_pos = end_idx
        
        pred_entities = pred_item['predictions']
        
        for j, token_span in enumerate(token_spans):
            gt_label = gt_item['labels'][j]
            
            # Find if this token overlaps with any predicted entity
            pred_label = "O"
            
            for entity in pred_entities:
                entity_span = (entity['start'], entity['end'])
                # strict overlap? or any overlap?
                # Using simple check: if token center is in entity? 
                # Or if intersection > 0
                if calculate_overlap(token_span, entity_span):
                    pred_label = entity['entity_group']
                    break
            
            y_true_all.append(gt_label)
            y_pred_all.append(pred_label)
            
            if gt_label == pred_label:
                correct_tokens += 1
            total_tokens += 1

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    print(f"Token-level Accuracy: {accuracy:.4f}")
    
    # Optional: Simple breakdown
    from collections import Counter
    # Helper to calculate F1 for a class
    def calc_f1(label):
        tp = sum(1 for t, p in zip(y_true_all, y_pred_all) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true_all, y_pred_all) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true_all, y_pred_all) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    # Get unique labels excluding O
    labels = set(y_true_all) | set(y_pred_all)
    if 'O' in labels:
        labels.remove('O')
        
    with open('evaluation_results.txt', 'w') as f:
        f.write(f"Token-level Accuracy: {accuracy:.4f}\n")
        f.write("\nPer-Class Metrics:\n")
        for label in sorted(labels):
            p, r, f1 = calc_f1(label)
            f.write(f"  {label}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}\n")
            print(f"  {label}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

if __name__ == "__main__":
    evaluate()
