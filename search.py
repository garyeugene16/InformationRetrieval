# --- Penjelasan Umum ---
# Skrip ini adalah antarmuka pengguna utama (front-end) untuk mini search engine.
# Pengguna dapat berinteraksi melalui command-line untuk:
# 1. Memilih model pencarian yang ingin digunakan (VSM atau BM25).
# 2. Memasukkan query pencarian.
# 3. Melihat hasil pencarian yang relevan, lengkap dengan skor dan cuplikan (snippet)
#    dari dokumen yang disesuaikan dengan query.
# 4. Menyimpan riwayat pencarian ke dalam file log.
# 5. Beralih antar model pencarian atau keluar dari program.

# --- Impor Pustaka ---
import json  # Untuk memuat data dokumen dari file.
from search_engine.vsm import VSMEngine  # Mengimpor kelas mesin pencari VSM.
from search_engine.bm25 import BM25Engine  # Mengimpor kelas mesin pencari BM25.
from search_engine.preprocessing import preprocess  # Mengimpor fungsi preprocessing untuk membersihkan query dan teks snippet.

# --- Pemuatan Dokumen ---
# Membuka dan memuat seluruh koleksi dokumen dari file JSON.
# Ini bisa diganti dengan 'data/documentsNew.json'
with open('data/documentsLibrary.json') as f:
    docs = json.load(f)

# --- Persiapan Data untuk Pencarian ---
# Mengekstrak teks mentah dari setiap dokumen untuk diindeks oleh mesin pencari.
raw_texts = [doc["text"] for doc in docs]
# Menyimpan ID dokumen dalam urutan yang sama untuk referensi nanti.
doc_ids = [doc["doc_id"] for doc in docs]
# Membuat 'lookup table' (seperti kamus) untuk mendapatkan teks asli dokumen dengan cepat menggunakan doc_id.
# Ini sangat efisien untuk mengambil teks saat akan membuat snippet.
doc_lookup = {doc["doc_id"]: doc["text"] for doc in docs}

# --- Inisialisasi Mesin Pencari ---
# Membuat instance dari VSMEngine dan BM25Engine.
# Pada tahap ini, seluruh dokumen dalam `raw_texts` akan di-preprocess dan di-indeks.
# Proses ini hanya dilakukan sekali di awal untuk efisiensi.
vsm_engine = VSMEngine(raw_texts)
bm25_engine = BM25Engine(raw_texts)

# --- Fungsi Pembuatan Snippet ---
def generate_snippet(query, doc_text, max_length=150):
    """
    Membuat cuplikan (snippet) dari teks dokumen yang relevan dengan query.
    Fungsi ini mencari kemunculan kata kunci dari query di dalam dokumen dan
    mengambil teks di sekitarnya untuk ditampilkan sebagai ringkasan.
    """

    # 1. Preprocess query untuk mendapatkan token-token penting dan menyimpannya dalam `set` untuk pencarian cepat.
    # Jika query kosong setelah diproses, kembalikan saja awal dokumen sebagai snippet.
    query_tokens = set(preprocess(query))
    if not query_tokens:
        return doc_text[:max_length] + "..."

    # 2. Pecah teks dokumen menjadi daftar kata-kata.
    words = doc_text.split()
    # Inisialisasi list untuk menyimpan posisi (indeks) di mana kata dari query ditemukan.
    match_positions = []

    # 3. Cari posisi kata dari query di dalam dokumen.
    for i, word in enumerate(words):
        # Setiap kata dari dokumen juga di-preprocess agar formatnya sama dengan token query (lowercase, stemmed).
        processed_word_tokens = preprocess(word)
        # Periksa apakah ada irisan (intersection) antara token query dengan token dari kata dokumen saat ini.
        if query_tokens.intersection(processed_word_tokens):
            match_positions.append(i)

    # 4. Jika tidak ada kata yang cocok sama sekali, ambil 20 kata pertama sebagai snippet.
    if not match_positions:
        return " ".join(words[:20]) + ("..." if len(words) > 20 else "")

    # 5. Jika ada kata yang cocok, ambil cuplikan di sekitar posisi *pertama* yang cocok.
    # Tentukan titik awal snippet, mundur 10 kata dari posisi kata pertama yang cocok.
    start = max(0, match_positions[0] - 10)
    # Tentukan titik akhir snippet, maju 25 kata dari titik awal untuk memberikan konteks.
    end = start + 25
    snippet = " ".join(words[start:end])

    # 6. Tambahkan elipsis (...) untuk menandakan bahwa snippet adalah potongan teks.
    # Tambahkan di awal jika snippet tidak dimulai dari kata pertama dokumen.
    if start > 0:
        snippet = "... " + snippet
    # Tambahkan di akhir jika snippet tidak berakhir di kata terakhir dokumen.
    if end < len(words):
        snippet = snippet + " ..."
        
    return snippet

# --- Fungsi Loop Pencarian Interaktif ---
def run_search_loop(searcher, engine_name):
    """
    Menjalankan loop utama untuk interaksi dengan pengguna.
    Menerima input query, menjalankan pencarian, dan menampilkan hasil.
    
    Args:
        searcher (object): Instance dari mesin pencari (bisa VSMEngine atau BM25Engine).
        engine_name (str): Nama engine yang sedang digunakan ('vsm' atau 'bm25').
        
    Returns:
        bool: False jika pengguna ingin keluar, True jika pengguna ingin mengganti engine.
    """

    while True:
        # Meminta input dari pengguna.
        query = input("\nMasukkan query ('exit' untuk keluar, 'engine' untuk ganti mesin): ")
        
        # Query untuk keluar dari program.
        if query.lower() == 'exit':
            return False
        # Query untuk kembali ke menu pemilihan engine.
        elif query.lower() == 'engine':
            return True

        # Menentukan jumlah hasil teratas yang akan ditampilkan.
        top_k = 5
        # Menjalankan pencarian menggunakan objek 'searcher' yang telah dipilih.
        ranked_ids, scores = searcher.search(query, top_k=top_k)

        # Memeriksa apakah ada hasil yang ditemukan.
        if not ranked_ids or all(scores[i] == 0 for i in ranked_ids):
            print("Tidak ada dokumen relevan ditemukan.")
            continue

        print(f"\nTop {top_k} dokumen relevan menurut {engine_name.upper()}:\n")
        
        # --- Menampilkan Hasil Pencarian ---
        # Iterasi melalui ID dokumen yang sudah diperingkat.
        for i in ranked_ids:
            # Mengambil ID dokumen aktual.
            doc_id = doc_ids[i]
            # Mengambil teks lengkap dokumen dari 'lookup table'.
            full_text = doc_lookup[doc_id]
            # Membuat snippet yang relevan dari teks lengkap tersebut.
            snippet = generate_snippet(query, full_text)
            # Mencetak hasil dalam format yang rapi.
            print(f"{doc_id} (Skor: {scores[i]:.4f}): {snippet}\n")

        # --- Menyimpan Hasil ke Log ---
        # Membuka file log dalam mode 'append' (`a`) dan dengan encoding utf-8.
        with open("result_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Query: {query} (Engine: {engine_name})\n")
            # Iterasi sekali lagi untuk menulis setiap hasil ke file log.
            for i in ranked_ids:
                doc_id = doc_ids[i]
                full_text = doc_lookup[doc_id]
                snippet = generate_snippet(query, full_text)
                f.write(f"{doc_id} (Skor: {scores[i]:.4f}): {snippet}\n")
            f.write("-" * 20 + "\n\n")

# --- Loop Program Utama ---
# Loop ini berjalan terus menerus sampai pengguna memilih untuk keluar.
while True:
    # Meminta pengguna untuk memilih mesin pencari.
    engine_choice = input("Pilih engine (vsm / bm25): ").lower()
    
    # Validasi input pengguna.
    if engine_choice not in ['vsm', 'bm25']:
        print("Pilihan tidak valid. Silakan pilih 'vsm' atau 'bm25'.")
        continue # Kembali meminta input.

    # Menentukan objek mesin pencari mana yang akan digunakan berdasarkan pilihan pengguna.
    searcher = vsm_engine if engine_choice == 'vsm' else bm25_engine
    
    # Menjalankan loop pencarian. `run_search_loop` akan mengembalikan nilai boolean.
    # Jika `False`, itu berarti pengguna mengetik 'exit'.
    if not run_search_loop(searcher, engine_choice):
        break # Keluar dari loop utama dan menghentikan program.
    # Jika `True`, loop utama akan berlanjut, memungkinkan pengguna memilih engine lagi.

print("Program selesai. Terima kasih!")