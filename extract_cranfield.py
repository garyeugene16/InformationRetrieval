# --- Penjelasan Umum ---
# Skrip ini bertanggung jawab untuk mengunduh dan memproses dataset standar dalam Information Retrieval,
# yaitu "Cranfield". Pustaka `ir_datasets` digunakan untuk mempermudah akses ke dataset ini.
# Hasilnya adalah dua file JSON: satu berisi koleksi dokumen dan satu lagi berisi
# query beserta daftar dokumen yang relevan (ground truth), yang akan digunakan untuk
# evaluasi mesin pencari.

# --- Impor Pustaka ---
import json  # Pustaka untuk bekerja dengan data format JSON (menyimpan hasil ke file).
import ir_datasets  # Pustaka khusus untuk mengakses dataset standar dalam riset Information Retrieval.

# --- Memuat Dataset ---
# Memuat dataset "cranfield" menggunakan pustaka ir_datasets.
# Objek 'dataset' ini menyediakan iterator untuk mengakses dokumen, query, dan qrels (relevance judgements).
dataset = ir_datasets.load("cranfield")

# --- 1. Ekstraksi Dokumen ---
# Bagian ini bertujuan untuk mengambil semua dokumen dari dataset Cranfield
# dan menyimpannya dalam format yang terstruktur.

# Menggunakan list comprehension untuk membuat daftar (list) dari semua dokumen.
# Setiap elemen dalam daftar adalah sebuah dictionary yang berisi 'doc_id' dan 'text'.
documents = [
    {"doc_id": doc.doc_id, "text": doc.text}
    for doc in dataset.docs_iter()  # Mengambil setiap dokumen dari iterator dokumen.
]

# Menyimpan daftar dokumen ke dalam file JSON.
# 'w' menandakan mode tulis (write). `with open(...)` memastikan file ditutup secara otomatis.
with open("data/documentsLibrary.json", "w") as f:

    # `json.dump` menulis objek Python (dalam hal ini, list of dictionaries) ke file `f`.
    # `indent=2` digunakan agar format file JSON rapi dan mudah dibaca manusia.
    json.dump(documents, f, indent=2)

# --- 2. Ekstraksi Query dan Penilaian Relevansi (Ground Truth) ---
# Bagian ini bertujuan untuk membuat file ground truth.
# Format akhirnya adalah sebuah dictionary di mana setiap key adalah teks query,
# dan value-nya adalah daftar (list) dari ID dokumen yang relevan untuk query tersebut.

# Membuat 'query_map', sebuah dictionary untuk memetakan ID query ke teks query-nya.
# Ini diperlukan karena file qrels (relevance judgements) hanya berisi ID, bukan teks lengkap.
query_map = {q.query_id: q.text for q in dataset.queries_iter()}

# Inisialisasi dictionary kosong untuk menyimpan hasil ground truth.
ground_truth = {}

# Iterasi melalui setiap data relevansi (qrel) dalam dataset.
# Setiap 'qrel' menghubungkan satu query_id dengan satu doc_id yang relevan.
for qrel in dataset.qrels_iter():

    # Menggunakan 'query_map' untuk mendapatkan teks query berdasarkan qrel.query_id.
    query_text = query_map[qrel.query_id]

    # Memeriksa apakah teks query ini sudah ada sebagai key di dictionary 'ground_truth'.
    if query_text not in ground_truth:
        # Jika belum ada, inisialisasi dengan list kosong.
        ground_truth[query_text] = []
        
    # Menambahkan doc_id yang relevan ke dalam list yang berasosiasi dengan teks query tersebut.
    ground_truth[query_text].append(qrel.doc_id)

# Menyimpan dictionary ground_truth ke dalam file JSON.
with open("data/ground_truthLibrary.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

# Memberikan pesan konfirmasi bahwa proses telah selesai dengan sukses.
print("Sukses ambil Cranfield dataset.")
