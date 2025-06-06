def precision_recall_f1(predicted, relevant):
    """
    Menghitung metrik evaluasi Precision, Recall, dan F1-Score.

    Args:
        predicted (list): Daftar ID dokumen yang dikembalikan oleh mesin pencari.
        relevant (list): Daftar ID dokumen yang sebenarnya relevan (ground truth).

    Returns:
        tuple: Sebuah tuple yang berisi (precision, recall, f1_score).
    """
    # Mengubah daftar (list) menjadi himpunan (set) untuk operasi irisan (intersection) yang efisien.
    predicted_set = set(predicted)
    relevant_set = set(relevant)
    
    # --- Menghitung True Positives (TP) ---
    # TP adalah jumlah dokumen yang ada di KEDUA himpunan:
    # diprediksi sebagai relevan DAN memang benar-benar relevan.
    # Operasi '&' pada set adalah operasi irisan.
    tp = len(predicted_set & relevant_set)
    
    # --- Menghitung Precision ---
    # Precision = (Jumlah dokumen relevan yang ditemukan) / (Total dokumen yang ditemukan)
    # Menjawab pertanyaan: "Dari semua yang dikembalikan, berapa persen yang benar?"
    # `if predicted else 0` digunakan untuk menghindari error pembagian dengan nol jika `predicted` kosong.
    precision = tp / len(predicted) if predicted else 0
    
    # --- Menghitung Recall ---
    # Recall = (Jumlah dokumen relevan yang ditemukan) / (Total dokumen relevan yang seharusnya ditemukan)
    # Menjawab pertanyaan: "Dari semua yang relevan, berapa persen yang berhasil ditemukan?"
    # `if relevant else 0` digunakan untuk menghindari error jika `relevant` (ground truth) kosong.
    recall = tp / len(relevant) if relevant else 0
    
    # --- Menghitung F1-Score ---
    # F1-Score adalah rata-rata harmonik dari Precision dan Recall.
    # Ini adalah metrik tunggal yang baik untuk menyeimbangkan antara precision dan recall.
    # `if (precision + recall) else 0` digunakan untuk menghindari error pembagian dengan nol.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    # Mengembalikan ketiga nilai metrik.
    return precision, recall, f1
