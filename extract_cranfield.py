import json
import ir_datasets

# Load dataset Cranfield
dataset = ir_datasets.load("cranfield")

# 1. Ekstrak dokumen
documents = [
    {"doc_id": doc.doc_id, "text": doc.text}
    for doc in dataset.docs_iter()
]

# Simpan ke documents.json
with open("data/documentsLibrary.json", "w") as f:
    json.dump(documents, f, indent=2)

# 2. Ekstrak query + relevansi (qrels)
# Format akhir: { query_text: [relevant_doc_ids] }
query_map = {q.query_id: q.text for q in dataset.queries_iter()}
ground_truth = {}

for qrel in dataset.qrels_iter():
    query_text = query_map[qrel.query_id]
    if query_text not in ground_truth:
        ground_truth[query_text] = []
    ground_truth[query_text].append(qrel.doc_id)

# Simpan ke ground_truth.json
with open("data/ground_truthLibrary.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

print("Sukses ambil Cranfield dataset.")
