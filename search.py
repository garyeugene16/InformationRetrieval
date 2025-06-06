import json
from search_engine.vsm import VSMEngine
from search_engine.bm25 import BM25Engine
from search_engine.preprocessing import preprocess # Diambil dari versi Hepi

# # --- Load Dokumen ---
# with open('data/documentsLibrary.json') as f:
#     docs = json.load(f)

# --- Load Dokumen ---
with open('data/documentsNew.json') as f:
    docs = json.load(f)

raw_texts = [doc["text"] for doc in docs]
doc_ids = [doc["doc_id"] for doc in docs]
doc_lookup = {doc["doc_id"]: doc["text"] for doc in docs}

# --- Inisialisasi Search Engine ---
vsm_engine = VSMEngine(raw_texts)
bm25_engine = BM25Engine(raw_texts)

# --- Fungsi Generate Snippet (Diambil dan diimprove dari versi Hepi) ---
def generate_snippet(query, doc_text, max_length=150):
    """
    Membuat cuplikan (snippet) dari teks dokumen yang relevan dengan query.
    """
    # Preprocess query untuk mendapatkan token-token penting
    query_tokens = set(preprocess(query))
    if not query_tokens:
        return doc_text[:max_length] + "..."

    words = doc_text.split()
    match_positions = []

    # Cari posisi kata dari query di dalam dokumen
    for i, word in enumerate(words):
        # Normalisasi kata dari dokumen sebelum membandingkan
        processed_word_tokens = preprocess(word)
        if query_tokens.intersection(processed_word_tokens):
            match_positions.append(i)

    # Jika tidak ada kata yang cocok, ambil awal dokumen sebagai snippet
    if not match_positions:
        return " ".join(words[:20]) + ("..." if len(words) > 20 else "")

    # Ambil cuplikan di sekitar posisi pertama yang cocok
    start = max(0, match_positions[0] - 10) # Mundur 10 kata dari kata pertama yang cocok
    end = start + 25 # Ambil sekitar 25 kata untuk konteks
    snippet = " ".join(words[start:end])

    # Tambahkan elipsis (...) jika snippet tidak dimulai dari awal atau tidak berakhir di akhir kalimat
    if start > 0:
        snippet = "... " + snippet
    if end < len(words):
        snippet = snippet + " ..."
        
    return snippet

# --- Fungsi Loop Pencarian (Struktur dari versi Gary, dimodifikasi untuk snippet) ---
def run_search_loop(searcher, engine_name):
    while True:
        query = input("\nMasukkan query ('exit' untuk keluar, 'engine' untuk ganti mesin): ")
        if query.lower() == 'exit':
            return False # Sinyal untuk menghentikan program sepenuhnya
        elif query.lower() == 'engine':
            return True # Sinyal untuk kembali ke menu pemilihan engine

        top_k = 5
        ranked_ids, scores = searcher.search(query, top_k=top_k)

        if not ranked_ids or all(scores[i] == 0 for i in ranked_ids):
            print("Tidak ada dokumen relevan ditemukan.")
            continue

        print(f"\nTop {top_k} dokumen relevan menurut {engine_name.upper()}:\n")
        
        # MENAMPILKAN HASIL DENGAN SNIPPET
        for i in ranked_ids:
            doc_id = doc_ids[i]
            # Hasilkan snippet untuk setiap hasil
            snippet = generate_snippet(query, doc_lookup[doc_id])
            print(f"{doc_id} (Skor: {scores[i]:.4f}): {snippet}\n")

        # Simpan hasil (TERMASUK SNIPPET) ke file log
        with open("result_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Query: {query} (Engine: {engine_name})\n")
            for i in ranked_ids:
                doc_id = doc_ids[i]
                snippet = generate_snippet(query, doc_lookup[doc_id])
                f.write(f"{doc_id} (Skor: {scores[i]:.4f}): {snippet}\n")
            f.write("-" * 20 + "\n\n")

# --- Loop Utama (Struktur dari versi Gary) ---
while True:
    engine_choice = input("Pilih engine (vsm / bm25): ").lower()
    
    if engine_choice not in ['vsm', 'bm25']:
        print("Pilihan tidak valid. Silakan pilih 'vsm' atau 'bm25'.")
        continue

    searcher = vsm_engine if engine_choice == 'vsm' else bm25_engine
    
    # Jalankan loop pencarian, dan periksa apakah user ingin ganti engine
    if not run_search_loop(searcher, engine_choice):
        break # Jika run_search_loop mengembalikan False (user ketik 'exit'), hentikan program.

print("Program selesai. Terima kasih!")