import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
import json
import os
from search_engine.preprocessing import preprocess #
from search_engine.vsm import VSMEngine #
from search_engine.bm25 import BM25Engine #
from search_engine.evaluation import precision_recall_f1 #

# --- Helper Function (di luar kelas utama) ---
def load_documents_from_json(file_path):
    """Memuat dokumen dari file JSON dan mengembalikan data atau pesan error."""
    docs, doc_names = [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i, item in enumerate(data):
                doc_name = str(item.get("id", f"Dokumen_{i+1}"))
                doc_content = item.get("text", "")
                if doc_content:
                    docs.append(doc_content)
                    doc_names.append(doc_name)
        if not docs:
            return None, None, "Tidak ada dokumen yang ditemukan dalam file JSON atau format tidak sesuai."
        return docs, doc_names, None
    except FileNotFoundError:
        return None, None, f"File JSON tidak ditemukan di: {file_path}"
    except json.JSONDecodeError:
        return None, None, f"Gagal membaca file JSON. Pastikan formatnya benar."
    except Exception as e:
        return None, None, f"Terjadi kesalahan: {e}"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Konfigurasi Window Utama ---
        self.title("Sistem Temu Kembali Informasi")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")  # "light", "dark", "system"
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Variabel State Aplikasi ---
        self.raw_docs = []
        self.doc_names = []
        self.engine = None
        self.predicted_ids = []

        # --- Frame Kiri (Kontrol) ---
        self.frame_left = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nsew")
        self.frame_left.grid_rowconfigure(8, weight=1) # Memberi ruang di bagian bawah

        # Widget di Frame Kiri
        self.label_title = ctk.CTkLabel(self.frame_left, text="Pengaturan", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_load = ctk.CTkButton(self.frame_left, text="Muat Dokumen", command=self.load_documents)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)
        
        self.label_status = ctk.CTkLabel(self.frame_left, text="Belum ada dokumen dimuat", text_color="gray")
        self.label_status.grid(row=2, column=0, padx=20, pady=(0, 20))

        self.model_var = ctk.StringVar(value="BM25")
        self.model_menu = ctk.CTkOptionMenu(self.frame_left, values=["BM25", "VSM"], variable=self.model_var, command=self.toggle_bm25_params)
        self.model_menu.grid(row=3, column=0, padx=20, pady=10)

        # Parameter BM25 (awal nya tersembunyi jika VSM dipilih, atau sebaliknya)
        self.k_slider = ctk.CTkSlider(self.frame_left, from_=0.0, to=3.0, number_of_steps=30)
        self.k_slider.set(1.5)
        self.k_label = ctk.CTkLabel(self.frame_left, text=f"BM25 k: {self.k_slider.get():.2f}")
        self.k_slider.configure(command=lambda v: self.k_label.configure(text=f"BM25 k: {v:.2f}"))
        self.k_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        self.k_slider.grid(row=5, column=0, padx=20, pady=(0, 10))

        self.b_slider = ctk.CTkSlider(self.frame_left, from_=0.0, to=1.0, number_of_steps=20)
        self.b_slider.set(0.75)
        self.b_label = ctk.CTkLabel(self.frame_left, text=f"BM25 b: {self.b_slider.get():.2f}")
        self.b_slider.configure(command=lambda v: self.b_label.configure(text=f"BM25 b: {v:.2f}"))
        self.b_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.b_slider.grid(row=7, column=0, padx=20, pady=(0, 20))

        self.toggle_bm25_params(self.model_var.get()) # Inisialisasi tampilan parameter
        
        # --- Frame Kanan (Query dan Hasil) ---
        self.frame_right = ctk.CTkFrame(self)
        self.frame_right.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_rowconfigure(1, weight=1) # Agar tabel hasil bisa membesar

        # Widget di Frame Kanan
        self.entry_query = ctk.CTkEntry(self.frame_right, placeholder_text="Masukkan kueri pencarian...")
        self.entry_query.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.btn_search = ctk.CTkButton(self.frame_right, text="Cari", width=100, command=self.search)
        self.btn_search.grid(row=0, column=1, sticky="e")

        self.setup_results_treeview()

        # --- Frame Evaluasi (di bawah tabel hasil) ---
        self.eval_frame = ctk.CTkFrame(self.frame_right)
        self.eval_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.eval_frame.grid_columnconfigure(1, weight=1)
        
        self.eval_label = ctk.CTkLabel(self.eval_frame, text="Evaluasi:", font=ctk.CTkFont(weight="bold"))
        self.eval_label.grid(row=0, column=0, padx=(10,0))
        
        self.entry_relevant = ctk.CTkEntry(self.eval_frame, placeholder_text="Masukkan ID relevan (indeks 0), pisahkan dengan koma. Cth: 0,2,5")
        self.entry_relevant.grid(row=0, column=1, sticky="ew", padx=10)

        self.btn_eval = ctk.CTkButton(self.eval_frame, text="Hitung", width=100, command=self.calculate_evaluation)
        self.btn_eval.grid(row=0, column=2, padx=(0,10), pady=10)

        self.label_precision = ctk.CTkLabel(self.eval_frame, text="Precision: -")
        self.label_precision.grid(row=1, column=0, columnspan=3, sticky="w", padx=10)
        self.label_recall = ctk.CTkLabel(self.eval_frame, text="Recall: -")
        self.label_recall.grid(row=2, column=0, columnspan=3, sticky="w", padx=10)
        self.label_f1 = ctk.CTkLabel(self.eval_frame, text="F1-Score: -")
        self.label_f1.grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=(0,10))

    def setup_results_treeview(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.configure("Treeview.Heading", background="#565b5e", foreground="white", font=('Calibri', 10,'bold'))
        style.map('Treeview.Heading', background=[('active','#3484F0')])

        self.tree = ttk.Treeview(self.frame_right, columns=("Peringkat", "ID Dokumen", "Skor", "Preview"), show="headings")
        self.tree.heading("Peringkat", text="Peringkat")
        self.tree.heading("ID Dokumen", text="ID Dokumen")
        self.tree.heading("Skor", text="Skor")
        self.tree.heading("Preview", text="Preview")

        self.tree.column("Peringkat", width=60, stretch=False)
        self.tree.column("ID Dokumen", width=150, stretch=False)
        self.tree.column("Skor", width=80, stretch=False)
        self.tree.column("Preview", width=400) # Biarkan ini meregang

        self.tree.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

    def load_documents(self):
        file_path = filedialog.askopenfilename(title="Pilih file JSON dataset", filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if not file_path:
            return

        self.raw_docs, self.doc_names, error = load_documents_from_json(file_path)
        if error:
            messagebox.showerror("Error Pemuatan Dokumen", error)
            return

        self.label_status.configure(text=f"{len(self.raw_docs)} dokumen dimuat", text_color="green")
        messagebox.showinfo("Sukses", f"{len(self.raw_docs)} dokumen berhasil dimuat.")
        self.initialize_engine()

    def initialize_engine(self):
        if not self.raw_docs:
            return
        
        model_type = self.model_var.get()
        if model_type == "BM25":
            k = self.k_slider.get()
            b = self.b_slider.get()
            self.engine = BM25Engine(self.raw_docs, k, b) #
        elif model_type == "VSM":
            self.engine = VSMEngine(self.raw_docs) #
        
        print(f"Engine {model_type} diinisialisasi.")

    def search(self):
        if not self.engine:
            messagebox.showwarning("Peringatan", "Harap muat dokumen dan inisialisasi engine terlebih dahulu.")
            return

        query = self.entry_query.get()
        if not query:
            messagebox.showwarning("Peringatan", "Harap masukkan kueri pencarian.")
            return

        # Hapus hasil lama dari treeview
        for i in self.tree.get_children():
            self.tree.delete(i)
        
        # Dapatkan hasil baru
        ranked_ids, scores = self.engine.search(query, top_k=len(self.raw_docs))
        self.predicted_ids = ranked_ids[:10] # Ambil 10 teratas untuk evaluasi
        
        # Tampilkan hasil baru
        for rank, doc_id in enumerate(ranked_ids[:20]): # Tampilkan 20 hasil teratas
            doc_name = self.doc_names[doc_id]
            score_val = f"{scores[doc_id]:.4f}"
            preview = self.raw_docs[doc_id][:100] + "..."
            self.tree.insert("", "end", values=(rank + 1, doc_name, score_val, preview))

    def toggle_bm25_params(self, choice):
        if choice == "BM25":
            self.k_label.grid()
            self.k_slider.grid()
            self.b_label.grid()
            self.b_slider.grid()
        else: # VSM
            self.k_label.grid_remove()
            self.k_slider.grid_remove()
            self.b_label.grid_remove()
            self.b_slider.grid_remove()
        
        if self.raw_docs: # Langsung inisialisasi ulang engine jika dokumen sudah dimuat
            self.initialize_engine()

    def calculate_evaluation(self):
        if not self.predicted_ids:
            messagebox.showwarning("Peringatan", "Lakukan pencarian terlebih dahulu untuk mendapatkan hasil prediksi.")
            return
        
        relevant_str = self.entry_relevant.get()
        if not relevant_str:
            messagebox.showwarning("Peringatan", "Masukkan ID dokumen relevan untuk evaluasi.")
            return
            
        try:
            relevant_ids = [int(id_str.strip()) for id_str in relevant_str.split(',')]
            
            p, r, f1 = precision_recall_f1(self.predicted_ids, relevant_ids) #
            
            self.label_precision.configure(text=f"Precision: {p:.4f}")
            self.label_recall.configure(text=f"Recall: {r:.4f}")
            self.label_f1.configure(text=f"F1-Score: {f1:.4f}")

        except ValueError:
            messagebox.showerror("Error", "Format ID relevan salah. Harap masukkan angka yang dipisahkan koma.")

if __name__ == "__main__":
    app = App()
    app.mainloop()