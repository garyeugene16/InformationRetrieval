import math
from .preprocessing import preprocess # Mengimpor fungsi preprocess untuk membersihkan teks
from collections import defaultdict, Counter # Mengimpor struktur data yang berguna

# Implementasi model pencarian Vector Space Model (VSM).
# VSM merepresentasikan dokumen dan kueri sebagai vektor dalam ruang multidimensi.
# Relevansi diukur berdasarkan jarak atau sudut antara vektor-vektor tersebut.
class VSMEngine:
    """
    Menginisialisasi engine Vector Space Model.

    Args:
        raw_docs (list): Daftar string, di mana setiap string adalah konten sebuah dokumen.
    """
    def __init__(self, raw_docs):
        # 1. Preprocessing: Mengubah setiap dokumen mentah menjadi daftar token.
        self.tokenized_docs = [preprocess(doc) for doc in raw_docs]
        self.doc_count = len(self.tokenized_docs)

        # --- Tahap Indexing ---
        # 2. Bangun Term Frequency (TF) dan Document Frequency (DF).
        self.tf_list = [] # Daftar untuk menyimpan Counter TF dari setiap dokumen.
        self.df = defaultdict(int) # Dictionary untuk menyimpan DF setiap term.
        for tokens in self.tokenized_docs:
            # Menghitung frekuensi term (tf) untuk dokumen saat ini.
            tf = Counter(tokens)
            self.tf_list.append(tf)
            # Memperbarui frekuensi dokumen (df) untuk setiap term unik dalam dokumen.
            for term in tf:
                self.df[term] += 1

        # Buat vocabulary dan term index
        # 3. Buat Vocabulary: Daftar unik dari semua term yang ada di koleksi.
        # Disortir agar urutan dimensi vektor selalu konsisten.
        self.vocab = sorted(self.df.keys())
        # Buat pemetaan dari term ke indeksnya dalam vocabulary.
        self.term_index = {term: idx for idx, term in enumerate(self.vocab)}

        # 4. Hitung Inverse Document Frequency (IDF) untuk setiap term di vocabulary.
        self.idf = {}
        for term in self.vocab:
            df = self.df[term]
            # Menggunakan "smoothed IDF" untuk menghindari pembagian dengan nol jika df = 0
            # dan untuk memberikan bobot pada term yang muncul di semua dokumen.
            self.idf[term] = math.log((self.doc_count) / (df + 1)) + 1  # smoothed idf

        # Buat matrix TF-IDF untuk dokumen
        # 5. Buat Vektor Dokumen: Mengubah setiap dokumen menjadi vektor TF-IDF.
        # Vektor-vektor ini akan disimpan untuk perbandingan saat pencarian.
        self.doc_vectors = [self._compute_vector(tf) for tf in self.tf_list]

    def _compute_vector(self, tf_counter):
        """
        Menghitung dan menormalisasi vektor TF-IDF untuk sebuah dokumen atau kueri.
        
        Args:
            tf_counter (Counter): Counter yang berisi frekuensi term (TF).
            
        Returns:
            list: Vektor TF-IDF yang sudah dinormalisasi (unit vector).
        """
        # Inisialisasi vektor dengan nol sebanyak ukuran vocabulary.
        vec = [0.0] * len(self.vocab)
        # Iterasi melalui setiap term dan frekuensinya di dalam input.
        for term, tf in tf_counter.items():
            # Hanya proses term yang ada di dalam vocabulary.
            if term in self.term_index:
                idx = self.term_index[term]
                # Pembobotan TF sublinear: mengurangi dampak dari frekuensi term yang sangat tinggi.
                # (1 + log(tf)) untuk tf > 0, dan 0 jika tf = 0.
                tf_weight = 1 + math.log(tf) if tf > 0 else 0  # sublinear tf
                # Hitung bobot TF-IDF dan tempatkan di posisi yang benar dalam vektor.
                vec[idx] = tf_weight * self.idf[term]
        
        # Normalisasi vektor agar menjadi unit vector (panjangnya 1).
        return self._normalize(vec)

    def _normalize(self, vec):
        """
        Menormalisasi sebuah vektor (L2 Normalization).
        
        Args:
            vec (list): Vektor yang akan dinormalisasi.
            
        Returns:
            list: Vektor yang sudah dinormalisasi.
        """
        # Hitung panjang Euclidean (L2 norm) dari vektor.
        norm = math.sqrt(sum(x ** 2 for x in vec))
        # Bagi setiap elemen vektor dengan normanya untuk mendapatkan unit vektor.
        # Menghindari pembagian dengan nol jika vektornya adalah vektor nol.
        return [x / norm for x in vec] if norm != 0 else vec

    def _cosine_similarity(self, vec1, vec2):
        """
        Menghitung Cosine Similarity antara dua vektor.
        Karena kedua vektor sudah dinormalisasi, ini sama dengan dot product.
        
        Args:
            vec1 (list): Vektor pertama (unit vector).
            vec2 (list): Vektor kedua (unit vector).
            
        Returns:
            float: Skor Cosine Similarity (antara 0 dan 1).
        """
        return sum(a * b for a, b in zip(vec1, vec2))

    def search(self, query, top_k=5):
        """
        Mencari dan memeringkat dokumen berdasarkan kueri yang diberikan.
        
        Args:
            query (str): String kueri dari pengguna.
            top_k (int): Jumlah dokumen teratas yang akan dikembalikan.
            
        Returns:
            tuple: Berisi daftar ID dokumen yang diperingkat dan daftar skor mentah.
        """
        # 1. Preprocess dan ubah kueri menjadi vektor TF-IDF.
        query_tokens = preprocess(query)
        query_tf = Counter(query_tokens)
        query_vec = self._compute_vector(query_tf)

        # 2. Hitung Cosine Similarity antara vektor kueri dan setiap vektor dokumen.
        scores = [self._cosine_similarity(query_vec, doc_vec) for doc_vec in self.doc_vectors]
        # 3. Peringkat dokumen: Mengurutkan indeks dokumen berdasarkan skornya (dari tertinggi ke terendah).
        ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # 4. Mengembalikan `top_k` dokumen teratas beserta skor mentahnya.
        return ranked_ids[:top_k], scores
