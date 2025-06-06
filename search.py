#GARYY
# import json
# from search_engine.vsm import VSMEngine
# from search_engine.bm25 import BM25Engine

# # Load documents
# with open('data/documentsLibrary.json') as f:
#     docs = json.load(f)

# with open('data/ground_truthLibrary.json') as f:
#     ground_truth = json.load(f)
    
# # # Load documents
# # with open('data/documents.json') as f:
# #     docs = json.load(f)

# # with open('data/ground_truth.json') as f:
# #     ground_truth = json.load(f)
    
# raw_texts = [doc["text"] for doc in docs]
# doc_ids = [doc["doc_id"] for doc in docs]
# doc_lookup = {doc["doc_id"]: doc["text"] for doc in docs}

# # Inisialisasi engine
# vsm_engine = VSMEngine(raw_texts)
# bm25_engine = BM25Engine(raw_texts)

# # Pilih engine
# engine = input("Pilih engine (vsm / bm25): ").lower()
# searcher = vsm_engine if engine == 'vsm' else bm25_engine

# # Loop pencarian
# while True:
#     query = input("\nMasukkan query (atau 'exit' untuk keluar): ")
#     if query.lower() == 'exit':
#         break

#     top_k = 5
#     ranked_ids, scores = searcher.search(query, top_k=top_k)
#     print(f"\nTop {top_k} dokumen relevan menurut {engine.upper()}:\n")
#     for i in ranked_ids:
#         doc_id = doc_ids[i]
#         print(f"{doc_id}: {doc_lookup[doc_id]}")
        # print(f"{doc_id}")

        
        
# import json
# from search_engine.vsm import VSMEngine
# from search_engine.bm25 import BM25Engine

# # Load documents
# with open('data/documents.json') as f:
#     docs = json.load(f)

# raw_texts = [doc["text"] for doc in docs]
# doc_ids = [doc["doc_id"] for doc in docs]
# doc_lookup = {doc["doc_id"]: doc["text"] for doc in docs}

# # Inisialisasi search engine
# vsm_engine = VSMEngine(raw_texts)
# bm25_engine = BM25Engine(raw_texts)

# def run_search_loop(searcher, engine_name):
#     while True:
#         query = input("\nMasukkan query ('exit' untuk keluar, 'engine' untuk ganti mesin): ")
#         if query.lower() == 'exit':
#             return False
#         elif query.lower() == 'engine':
#             return True

#         top_k = 5
#         ranked_ids, scores = searcher.search(query, top_k=top_k)

#         if not ranked_ids or all(scores[i] == 0 for i in ranked_ids):
#             print("Tidak ada dokumen relevan ditemukan.")
#             continue

#         print(f"\nTop {top_k} dokumen relevan menurut {engine_name.upper()}:\n")
#         for i in ranked_ids:
#             doc_id = doc_ids[i]
#             print(f"{doc_id} (Skor: {scores[i]:.4f}): {doc_lookup[doc_id]}\n")

#         # Simpan hasil ke file log
#         with open("result_log.txt", "a") as f:
#             f.write(f"Query: {query} (Engine: {engine_name})\n")
#             for i in ranked_ids:
#                 doc_id = doc_ids[i]
#                 f.write(f"{doc_id} (Skor: {scores[i]:.4f}): {doc_lookup[doc_id]}\n")
#             f.write("\n")

# # Main loop (ganti engine jika diminta)
# while True:
#     engine = input("Pilih engine (vsm / bm25): ").lower()
#     searcher = vsm_engine if engine == 'vsm' else bm25_engine
#     repeat = run_search_loop(searcher, engine)
#     if not repeat:
#         break






# #====================HEPI=====================
import json
from search_engine.vsm import VSMEngine
from search_engine.bm25 import BM25Engine
from search_engine.preprocessing import preprocess

# Load documents
with open('data/documentsLibrary.json') as f:
    docs = json.load(f)

raw_texts = [doc["text"] for doc in docs]
doc_ids = [doc["doc_id"] for doc in docs]
doc_lookup = {doc["doc_id"]: doc["text"] for doc in docs}

# Inisialisasi engine
vsm_engine = VSMEngine(raw_texts)
bm25_engine = BM25Engine(raw_texts)

# Fungsi untuk membuat cuplikan
def generate_snippet(query, doc_text, max_length=100):
    query_tokens = set(preprocess(query))  # Token query yang sudah diproses
    words = doc_text.split()
    # Cari posisi kata query dalam dokumen
    match_positions = []
    for i, word in enumerate(words):
        processed_words = preprocess(word)  # Preprocess kata dari dokumen
        for processed_word in processed_words:  # Iterasi setiap token hasil preprocessing
            if processed_word in query_tokens:
                match_positions.append(i)
                break  # Hentikan iterasi jika menemukan kecocokan
    if not match_positions:
        # Jika tidak ada kata yang cocok, ambil awal dokumen
        snippet = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
        return snippet[:max_length]
    
    # Ambil cuplikan di sekitar posisi pertama yang cocok
    start = max(0, match_positions[0] - 5)
    end = start + 10
    snippet = " ".join(words[start:end]) + ("..." if end < len(words) else "")
    return snippet[:max_length]

# Pilih engine
engine = input("Pilih engine (vsm / bm25): ").lower()
searcher = vsm_engine if engine == 'vsm' else bm25_engine

# Loop pencarian
while True:
    query = input("\nMasukkan query (atau 'exit' untuk keluar): ")
    if query.lower() == 'exit':
        break

    top_k = 5
    ranked_ids, scores = searcher.search(query, top_k=top_k)
    print(f"\nTop {top_k} dokumen relevan menurut {engine.upper()}:\n")
    for i, idx in enumerate(ranked_ids):
        doc_id = doc_ids[idx]
        score = scores[idx]
        snippet = generate_snippet(query, doc_lookup[doc_id])
        print(f"{doc_id} (Score: {score:.4f}): {snippet}")