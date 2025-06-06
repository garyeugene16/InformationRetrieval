# --- Penjelasan Umum ---
# Skrip ini berfungsi sebagai 'benchmark' atau pengujian kuantitatif untuk membandingkan
# performa dari dua model mesin pencari: Vector Space Model (VSM) dan Okapi BM25.
# Prosesnya adalah sebagai berikut:
# 1. Memuat koleksi dokumen dan 'ground truth' (kunci jawaban query dan dokumen relevannya).
# 2. Menginisialisasi kedua mesin pencari (VSM dan BM25) dengan koleksi dokumen.
# 3. Untuk setiap query dalam ground truth, skrip akan:
#    a. Menjalankan pencarian menggunakan VSM dan BM25.
#    b. Membandingkan hasil pencarian dengan daftar dokumen relevan dari ground truth.
#    c. Menghitung metrik evaluasi: Precision, Recall, dan F1-score.
# 4. Menghitung dan menampilkan rata-rata dari semua metrik untuk memberikan kesimpulan
#    model mana yang berkinerja lebih baik secara keseluruhan pada dataset ini.
# 5. Secara opsional, menampilkan visualisasi perbandingan dalam bentuk diagram batang.

# --- Impor Pustaka ---
import json  # Untuk memuat data dari file JSON.
from search_engine.vsm import VSMEngine  # Mengimpor kelas mesin pencari VSM.
from search_engine.bm25 import BM25Engine  # Mengimpor kelas mesin pencari BM25.
from search_engine.evaluation import precision_recall_f1  # Mengimpor fungsi untuk menghitung metrik evaluasi.
import matplotlib.pyplot as plt  # Pustaka untuk membuat plot/grafik visualisasi, bersifat opsional.

# --- Pemuatan Data ---
# Memuat data dokumen dari file JSON yang telah disiapkan oleh `extract_cranfield.py`.
# Ini bisa diganti juga dengan 'data/documentsNew.json'
with open('data/documentsLibrary.json') as f:
    docs = json.load(f)

# Memuat data ground truth (query dan dokumen relevan) dari file JSON.
with open('data/ground_truthLibrary.json') as f:
    ground_truth = json.load(f)

# --- Persiapan Data ---
# Mengekstrak hanya teks mentah dari setiap objek dokumen ke dalam list `raw_texts`.
# List ini akan menjadi input untuk mesin pencari.
raw_texts = [doc["text"] for doc in docs]

# Mengekstrak ID dokumen ke dalam list `doc_ids`. Urutannya sama dengan `raw_texts`.
# Ini penting untuk mengubah hasil pencarian (yang berupa indeks) kembali menjadi ID dokumen.
doc_ids = [doc["doc_id"] for doc in docs]

# --- Inisialisasi Mesin Pencari ---
# Membuat instance dari VSMEngine. Proses ini akan mencakup preprocessing dan indexing seluruh dokumen.
vsm_engine = VSMEngine(raw_texts)

# Membuat instance dari BM25Engine. Proses ini juga mencakup preprocessing dan indexing.
bm25_engine = BM25Engine(raw_texts)

print("Evaluasi Mini Search Engine dengan VSM dan BM25\n")

# --- Penyimpanan Metrik ---
# Menyiapkan dictionary untuk menyimpan hasil metrik dari setiap query.
# Ini akan digunakan untuk menghitung rata-rata performa di akhir.
vsm_metrics = {'precision': [], 'recall': [], 'f1': []}
bm25_metrics = {'precision': [], 'recall': [], 'f1': []}

# --- Loop Evaluasi Utama ---
# Iterasi melalui setiap pasangan (query, daftar_dokumen_relevan) dalam ground_truth.
for query, relevant_docs in ground_truth.items():
    # --- Pencarian dengan VSM ---
    # Menjalankan pencarian. `vsm_engine.search` mengembalikan indeks dokumen yang diperingkat.
    # Skor mentah (variabel kedua, `_`) tidak digunakan dalam evaluasi ini.
    vsm_indices, _ = vsm_engine.search(query)
    # Mengubah hasil indeks menjadi ID dokumen yang sebenarnya menggunakan `doc_ids`.
    vsm_doc_ids = [doc_ids[i] for i in vsm_indices]

    # --- Pencarian dengan BM25 ---
    # Proses yang sama diulang untuk mesin pencari BM25.
    bm25_indices, _ = bm25_engine.search(query)
    bm25_doc_ids = [doc_ids[i] for i in bm25_indices]

    # --- Perhitungan Metrik ---
    # Menghitung Precision, Recall, dan F1-score untuk VSM dengan membandingkan hasil prediksi (`vsm_doc_ids`)
    # dengan ground truth (`relevant_docs`).
    vsm_prec, vsm_rec, vsm_f1 = precision_recall_f1(vsm_doc_ids, relevant_docs)
    # Proses yang sama diulang untuk BM25.
    bm25_prec, bm25_rec, bm25_f1 = precision_recall_f1(bm25_doc_ids, relevant_docs)

    # --- Menyimpan Metrik per Query---
    # Menambahkan hasil metrik dari query saat ini ke dalam list masing-masing.
    vsm_metrics['precision'].append(vsm_prec)
    vsm_metrics['recall'].append(vsm_rec)
    vsm_metrics['f1'].append(vsm_f1)
    bm25_metrics['precision'].append(bm25_prec)
    bm25_metrics['recall'].append(bm25_rec)
    bm25_metrics['f1'].append(bm25_f1)

    # Menampilkan hasil evaluasi untuk query saat ini ke layar.
    print(f"Query: {query}")
    print(f"  VSM   → Precision: {vsm_prec:.2f}, Recall: {vsm_rec:.2f}, F1-score: {vsm_f1:.2f}")
    print(f"  BM25  → Precision: {bm25_prec:.2f}, Recall: {bm25_rec:.2f}, F1-score: {bm25_f1:.2f}\n")

# --- Perhitungan Rata-rata Metrik ---
# Setelah loop selesai, hitung rata-rata untuk setiap metrik pada kedua model.
vsm_avg = {k: sum(v) / len(v) for k, v in vsm_metrics.items()}
bm25_avg = {k: sum(v) / len(v) for k, v in bm25_metrics.items()}

# --- Menampilkan Hasil Akhir ---
# Mencetak rata-rata metrik ke layar.
print("Rata-rata Metrik:")
print(f"  VSM   → Precision: {vsm_avg['precision']:.2f}, Recall: {vsm_avg['recall']:.2f}, F1-score: {vsm_avg['f1']:.2f}")
print(f"  BM25  → Precision: {bm25_avg['precision']:.2f}, Recall: {bm25_avg['recall']:.2f}, F1-score: {bm25_avg['f1']:.2f}\n")

# --- Kesimpulan Perbandingan ---
# Membuat kesimpulan otomatis berdasarkan perbandingan F1-score rata-rata.
print("Kesimpulan:")
if vsm_avg['f1'] > bm25_avg['f1']:
    print("VSM memiliki performa lebih baik berdasarkan nilai rata-rata F1-score.")
elif bm25_avg['f1'] > vsm_avg['f1']:
    print("BM25 memiliki performa lebih baik berdasarkan nilai rata-rata F1-score.")
else:
    print("VSM dan BM25 memiliki performa seimbang berdasarkan nilai rata-rata F1-score.")
# Menambahkan analisis kualitatif singkat tentang kekuatan masing-masing model.
print("BM25 cenderung lebih robust karena mempertimbangkan panjang dokumen dan frekuensi kata, sedangkan VSM lebih sederhana dan efektif untuk dokumen dengan panjang seragam.")

# --- Visualisasi (Opsional) ---
# Bagian ini akan membuat diagram batang jika pustaka matplotlib terinstal.
# Jika tidak, ia akan mencetak pesan dan melanjutkan tanpa error.
try:
    # Menyiapkan data dan label untuk plot.
    metrics = ['Precision', 'Recall', 'F1-score']
    vsm_values = [vsm_avg['precision'], vsm_avg['recall'], vsm_avg['f1']]
    bm25_values = [bm25_avg['precision'], bm25_avg['recall'], bm25_avg['f1']]

    # Membuat plot diagram batang untuk perbandingan.
    x = range(len(metrics))
    plt.bar(x, vsm_values, width=0.4, label='VSM', color='skyblue')
    plt.bar([i + 0.4 for i in x], bm25_values, width=0.4, label='BM25', color='salmon')

    # Menambahkan label dan judul pada plot.
    plt.xlabel('Metrik')
    plt.ylabel('Nilai')
    plt.title('Perbandingan Metrik VSM dan BM25')
    plt.xticks([i + 0.2 for i in x], metrics) # Atur posisi label sumbu-x di tengah antara dua batang.
    plt.legend()

    # Menampilkan plot ke layar.
    plt.show()
except ImportError:
    # Menangkap error jika matplotlib tidak ditemukan dan memberikan pesan informatif.
    print("Visualisasi tidak ditampilkan karena matplotlib tidak terinstal.")