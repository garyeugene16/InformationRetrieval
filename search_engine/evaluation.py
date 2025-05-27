def precision_recall_f1(predicted, relevant):
    predicted_set = set(predicted)
    relevant_set = set(relevant)
    tp = len(predicted_set & relevant_set)
    precision = tp / len(predicted) if predicted else 0
    recall = tp / len(relevant) if relevant else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1
