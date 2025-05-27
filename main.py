# import json
# from search_engine.vsm import VSMEngine
# from search_engine.bm25 import BM25Engine
# from search_engine.evaluation import precision_recall_f1

# # Load documents
# with open('data/documents.json') as f:
#     docs = json.load(f)

# with open('data/ground_truth.json') as f:
#     ground_truth = json.load(f)

# raw_texts = [doc["content"] for doc in docs]
# queries = ground_truth["queries"]

# vsm_engine = VSMEngine(raw_texts)
# bm25_engine = BM25Engine(raw_texts)

# print("Evaluasi Mini Search Engine dengan VSM dan BM25\n")
# for q in queries:
#     query_text = q["query_text"]
#     true_ids = q["relevant_docs"]

#     vsm_result, _ = vsm_engine.search(query_text)
#     bm25_result, _ = bm25_engine.search(query_text)

#     vsm_prec, vsm_rec, vsm_f1 = precision_recall_f1(vsm_result, true_ids)
#     bm25_prec, bm25_rec, bm25_f1 = precision_recall_f1(bm25_result, true_ids)

#     print(f"Query: {query_text}")
#     print(f"  VSM   → P: {vsm_prec:.2f}, R: {vsm_rec:.2f}, F1: {vsm_f1:.2f}")
#     print(f"  BM25  → P: {bm25_prec:.2f}, R: {bm25_rec:.2f}, F1: {bm25_f1:.2f}")


# #GARY FINAL
# import json
# from search_engine.vsm import VSMEngine
# from search_engine.bm25 import BM25Engine
# from search_engine.evaluation import precision_recall_f1

# # Load documents dan ground truth
# with open('data/documents.json') as f:
#     docs = json.load(f)

# with open('data/ground_truth.json') as f:
#     ground_truth = json.load(f)
    
# # # Load documents dan ground truth
# # with open('data/documentsLibrary.json') as f:
# #     docs = json.load(f)

# # with open('data/ground_truthLibrary.json') as f:
# #     ground_truth = json.load(f)

# # Ambil teks dokumen dan buat mapping indeks ke doc_id
# raw_texts = [doc["text"] for doc in docs]
# doc_ids = [doc["doc_id"] for doc in docs]

# # Inisialisasi engine
# vsm_engine = VSMEngine(raw_texts)
# bm25_engine = BM25Engine(raw_texts)

# print("Evaluasi Mini Search Engine dengan VSM dan BM25\n")

# for query, relevant_docs in ground_truth.items():
#     # Cari dengan VSM
#     vsm_indices, _ = vsm_engine.search(query)
#     vsm_doc_ids = [doc_ids[i] for i in vsm_indices]

#     # Cari dengan BM25
#     bm25_indices, _ = bm25_engine.search(query)
#     bm25_doc_ids = [doc_ids[i] for i in bm25_indices]

#     # Hitung metric evaluasi
#     vsm_prec, vsm_rec, vsm_f1 = precision_recall_f1(vsm_doc_ids, relevant_docs)
#     bm25_prec, bm25_rec, bm25_f1 = precision_recall_f1(bm25_doc_ids, relevant_docs)

#     print(f"Query: {query}")
#     print(f"  VSM   → Precision: {vsm_prec:.5f}, Recall: {vsm_rec:.5f}, F1-score: {vsm_f1:.5f}")
#     print(f"  BM25  → Precision: {bm25_prec:.5f}, Recall: {bm25_rec:.5f}, F1-score: {bm25_f1:.5f}\n")

import json
from search_engine.vsm import VSMEngine
from search_engine.bm25 import BM25Engine
from search_engine.evaluation import precision_recall_f1
import matplotlib.pyplot as plt  # Opsional, untuk visualisasi

# Load documents dan ground truth
with open('data/documentsLibrary.json') as f:
    docs = json.load(f)

with open('data/ground_truthLibrary.json') as f:
    ground_truth = json.load(f)

# Ambil teks dokumen dan buat mapping indeks ke doc_id
raw_texts = [doc["text"] for doc in docs]
doc_ids = [doc["doc_id"] for doc in docs]

# Inisialisasi engine
vsm_engine = VSMEngine(raw_texts)
bm25_engine = BM25Engine(raw_texts)

print("Evaluasi Mini Search Engine dengan VSM dan BM25\n")

# Simpan metrik untuk perbandingan
vsm_metrics = {'precision': [], 'recall': [], 'f1': []}
bm25_metrics = {'precision': [], 'recall': [], 'f1': []}

# Evaluasi untuk setiap query di ground truth
for query, relevant_docs in ground_truth.items():
    # Cari dengan VSM
    vsm_indices, _ = vsm_engine.search(query)
    vsm_doc_ids = [doc_ids[i] for i in vsm_indices]

    # Cari dengan BM25
    bm25_indices, _ = bm25_engine.search(query)
    bm25_doc_ids = [doc_ids[i] for i in bm25_indices]

    # Hitung metrik evaluasi
    vsm_prec, vsm_rec, vsm_f1 = precision_recall_f1(vsm_doc_ids, relevant_docs)
    bm25_prec, bm25_rec, bm25_f1 = precision_recall_f1(bm25_doc_ids, relevant_docs)

    # Simpan metrik
    vsm_metrics['precision'].append(vsm_prec)
    vsm_metrics['recall'].append(vsm_rec)
    vsm_metrics['f1'].append(vsm_f1)
    bm25_metrics['precision'].append(bm25_prec)
    bm25_metrics['recall'].append(bm25_rec)
    bm25_metrics['f1'].append(bm25_f1)

    # Tampilkan hasil per query
    print(f"Query: {query}")
    print(f"  VSM   → Precision: {vsm_prec:.2f}, Recall: {vsm_rec:.2f}, F1-score: {vsm_f1:.2f}")
    print(f"  BM25  → Precision: {bm25_prec:.2f}, Recall: {bm25_rec:.2f}, F1-score: {bm25_f1:.2f}\n")

# Hitung rata-rata metrik
vsm_avg = {k: sum(v) / len(v) for k, v in vsm_metrics.items()}
bm25_avg = {k: sum(v) / len(v) for k, v in bm25_metrics.items()}

# Tampilkan rata-rata metrik
print("Rata-rata Metrik:")
print(f"  VSM   → Precision: {vsm_avg['precision']:.2f}, Recall: {vsm_avg['recall']:.2f}, F1-score: {vsm_avg['f1']:.2f}")
print(f"  BM25  → Precision: {bm25_avg['precision']:.2f}, Recall: {bm25_avg['recall']:.2f}, F1-score: {bm25_avg['f1']:.2f}\n")

# Kesimpulan
print("Kesimpulan:")
if vsm_avg['f1'] > bm25_avg['f1']:
    print("VSM memiliki performa lebih baik berdasarkan nilai rata-rata F1-score.")
elif bm25_avg['f1'] > bm25_avg['f1']:
    print("BM25 memiliki performa lebih baik berdasarkan nilai rata-rata F1-score.")
else:
    print("VSM dan BM25 memiliki performa seimbang berdasarkan nilai rata-rata F1-score.")
print("BM25 cenderung lebih robust karena mempertimbangkan panjang dokumen dan frekuensi kata, sedangkan VSM lebih sederhana dan efektif untuk dokumen dengan panjang seragam.")

# Visualisasi (Opsional)
try:
    metrics = ['Precision', 'Recall', 'F1-score']
    vsm_values = [vsm_avg['precision'], vsm_avg['recall'], vsm_avg['f1']]
    bm25_values = [bm25_avg['precision'], bm25_avg['recall'], bm25_avg['f1']]

    x = range(len(metrics))
    plt.bar(x, vsm_values, width=0.4, label='VSM', color='skyblue')
    plt.bar([i + 0.4 for i in x], bm25_values, width=0.4, label='BM25', color='salmon')
    plt.xlabel('Metrik')
    plt.ylabel('Nilai')
    plt.title('Perbandingan Metrik VSM dan BM25')
    plt.xticks([i + 0.2 for i in x], metrics)
    plt.legend()
    plt.show()
except ImportError:
    print("Visualisasi tidak ditampilkan karena matplotlib tidak terinstal.")