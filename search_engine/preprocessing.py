# Mengimpor library yang diperlukan
import re # 're' untuk operasi Regular Expression, digunakan untuk membersihkan teks.
import nltk # 'nltk' (Natural Language Toolkit) adalah library utama untuk pemrosesan bahasa.
from nltk.corpus import stopwords # Mengimpor daftar kata-kata umum (stopwords) dari NLTK.
from nltk.stem import PorterStemmer # Mengimpor algoritma Porter Stemmer untuk mengubah kata ke bentuk dasarnya.

# Persiapan Awal (dilakukan sekali saat program pertama kali dijalankan)
# Mengunduh paket 'stopwords' dari NLTK jika belum terinstal di lingkungan Anda.
# Ini berisi daftar stopwords untuk berbagai bahasa.
nltk.download('stopwords')

# Memuat daftar stopwords untuk bahasa Inggris dan menyimpannya sebagai 'set'.
# Menggunakan 'set' membuat proses pemeriksaan stopword menjadi sangat cepat.
stop_words = set(stopwords.words('english'))

# Membuat instance dari kelas PorterStemmer.
# Objek ini akan digunakan untuk melakukan stemming pada kata-kata.
stemmer = PorterStemmer()

def preprocess(text):
    # Lowercase, remove non-alphabet
    """
    Fungsi untuk melakukan preprocessing pada sebuah string teks.
    Proses ini mencakup:
    1. Case Folding (mengubah ke huruf kecil).
    2. Cleaning (menghapus karakter selain huruf).
    3. Tokenization (memecah teks menjadi kata-kata/token).
    4. Stopword Removal (menghapus kata-kata umum).
    5. Stemming (mengubah kata ke bentuk dasarnya).

    Args:
        text (str): String teks mentah yang akan diproses.

    Returns:
        list: Sebuah daftar (list) berisi token-token yang sudah bersih dan diproses.
    """
    # 1: Case Folding dan Cleaning
    # Mengubah semua teks menjadi huruf kecil (lowercase).
    # re.sub(r'[^a-zA-Z\s]', '', ...) menghapus semua karakter yang BUKAN (^)
    # huruf (a-z, A-Z) atau spasi (\s).
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # 2: Tokenization
    # Memecah string yang sudah bersih menjadi daftar token berdasarkan spasi.
    tokens = text.split()
    
    # 3: Stopword Removal & Stemming (dilakukan dalam satu langkah menggunakan list comprehension)
    # Untuk setiap 'word' dalam 'tokens':
    # - Cek dulu apakah 'word' BUKAN stopword (`if word not in stop_words`).
    # - Jika bukan stopword, ubah kata tersebut ke bentuk dasarnya menggunakan stemmer (`stemmer.stem(word)`).
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    # Mengembalikan daftar token yang sudah selesai diproses
    return tokens
