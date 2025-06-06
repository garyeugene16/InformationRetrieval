import math
from .preprocessing import preprocess # Mengimpor fungsi preprocess untuk membersihkan teks
from collections import defaultdict, Counter # Mengimpor struktur data


class BM25Engine:
    """
    Menginisialisasi engine BM25.
    
    Args:
        raw_docs (list): Daftar string, di mana setiap string adalah konten sebuah dokumen.
        k (float): Parameter BM25 untuk saturasi frekuensi kata. Nilai umum antara 1.2 dan 2.0.
        b (float): Parameter BM25 untuk normalisasi panjang dokumen. Nilai umum adalah 0.75.
    """
    def __init__(self, raw_docs, k=1.5, b=0.75):
        # Menyimpan parameter BM25
        self.k = k
        self.b = b

        # Preprocessing
        # --- Tahap Preprocessing dan Indexing ---
        # 1. Tokenisasi: Mengubah setiap dokumen mentah menjadi daftar token (kata-kata) yang sudah dibersihkan.
        self.tokenized_docs = [preprocess(doc) for doc in raw_docs]
        # 2. Menghitung dan menyimpan panjang setiap dokumen setelah tokenisasi.
        self.doc_lens = [len(doc) for doc in self.tokenized_docs]
        # 3. Menghitung panjang rata-rata dari semua dokumen dalam koleksi.
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens)

         # 4. Inisialisasi struktur data untuk statistik korpus.
        # df: Document Frequency -> Berapa banyak dokumen yang mengandung sebuah term.
        self.doc_freqs = defaultdict(int)   # df: berapa dokumen mengandung term
        # tf: Term Frequency -> Daftar berisi Counter untuk setiap dokumen, menyimpan frekuensi setiap term.
        self.term_freqs = []                # tf: list of Counter per dokumen

         # 5. Proses indexing: Menghitung df dan tf untuk seluruh koleksi.
        for tokens in self.tokenized_docs:
            # Menghitung frekuensi term (tf) untuk dokumen saat ini.
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            # Memperbarui frekuensi dokumen (df) untuk setiap term unik dalam dokumen.
            for term in tf:
                self.doc_freqs[term] += 1

        # Menyimpan jumlah total dokumen dalam koleksi.
        self.N = len(self.tokenized_docs)  # jumlah dokumen

    def idf(self, term):
        # Inverse Document Frequency (menggunakan formula BM25)
        """
        Menghitung skor Inverse Document Frequency (IDF) untuk sebuah term.
        Formula IDF yang digunakan di sini adalah varian spesifik dari BM25.
        
        Args:
            term (str): Term yang akan dihitung IDF-nya.
            
        Returns:
            float: Skor IDF dari term tersebut.
        """
        # Mengambil frekuensi dokumen (df) dari term. Jika term tidak ada, df = 0.
        df = self.doc_freqs.get(term, 0)
        # Jika term tidak pernah muncul di dokumen manapun, skor IDF-nya 0.
        if df == 0:
            return 0
        # Formula IDF BM25. Memberikan skor lebih tinggi untuk term yang lebih langka.
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens, doc_idx):
        """
        Menghitung skor relevansi BM25 untuk satu dokumen terhadap sebuah kueri.
        
        Args:
            query_tokens (list): Daftar token dari kueri yang sudah diproses.
            doc_idx (int): Indeks dari dokumen yang akan dihitung skornya.
            
        Returns:
            float: Skor BM25.
        """
        score = 0
        # Mengambil data tf dan panjang untuk dokumen target.
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        # Iterasi melalui setiap term dalam kueri untuk mengakumulasi skor.
        for term in query_tokens:
            # Jika term dari kueri tidak ada di dalam dokumen, lewati (tidak ada kontribusi skor).
            if term not in doc_tf:
                continue
            
            # Mengambil frekuensi term (tf) di dalam dokumen.
            tf = doc_tf[term]
            # Menghitung IDF untuk term tersebut.
            idf = self.idf(term)
            # --- Formula Inti BM25 untuk satu term ---
            # Numerator: Memberi bobot pada frekuensi term (tf), dikontrol oleh parameter k.
            numerator = tf * (self.k + 1)
            # Denominator: Menormalisasi bobot tf berdasarkan panjang dokumen, dikontrol oleh parameter k dan b.
            denominator = tf + self.k * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            # Akumulasi skor dari term ini ke skor total dokumen.
            score += idf * (numerator / denominator)
        return score

    def search(self, query, top_k=5):
        """
        Mencari dan memeringkat dokumen berdasarkan kueri yang diberikan.
        
        Args:
            query (str): String kueri dari pengguna.
            top_k (int): Jumlah dokumen teratas yang akan dikembalikan.
            
        Returns:
            tuple: Berisi daftar ID dokumen yang diperingkat dan daftar skor mentah.
        """
        # 1. Preprocess kueri pengguna.
        query_tokens = preprocess(query)
        # 2. Hitung skor BM25 untuk setiap dokumen dalam koleksi.
        scores = [self.score(query_tokens, i) for i in range(len(self.tokenized_docs))]
        # 3. Peringkat dokumen: Mengurutkan indeks dokumen berdasarkan skornya (dari tertinggi ke terendah).
        ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # 4. Mengembalikan `top_k` dokumen teratas beserta skor mentahnya.
        return ranked_ids[:top_k], scores
