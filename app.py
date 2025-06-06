import json
import streamlit as st
import pandas as pd # Import Pandas
from search_engine.preprocessing import preprocess #
from search_engine.vsm import VSMEngine #
from search_engine.bm25 import BM25Engine #
from search_engine.evaluation import precision_recall_f1 #
import os

# --- Tentukan nama subfolder data dan nama file dataset Anda ---
DATA_SUBFOLDER = "data"
DATASET_FILENAME = "documentsLibrary.json" # Pastikan nama file ini sesuai

# --- Dapatkan path absolut ke direktori tempat app.py berada (project_root) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Gabungkan path direktori dengan nama subfolder dan nama file ---
STATIC_JSON_FILE_PATH = os.path.join(APP_DIR, DATA_SUBFOLDER, DATASET_FILENAME)

# --- Inisialisasi st.session_state di awal ---
if 'raw_docs' not in st.session_state:
    st.session_state.raw_docs = []
    st.session_state.doc_names = []
    st.session_state.engine = None
    st.session_state.predicted_ids_for_eval = []
    st.session_state.current_engine_key = None

# --- Cache untuk performa ---
@st.cache_resource
def load_engine(engine_type, docs, k=1.5, b=0.75):
    if engine_type == "BM25":
        return BM25Engine(docs, k, b) #
    elif engine_type == "VSM":
        return VSMEngine(docs) #
    return None

def load_documents_from_json(file_path):
    docs = []
    doc_names = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for i, item in enumerate(data): # Tambahkan enumerate untuk ID default yang lebih baik
                # Sesuaikan 'item.get("id")' dan 'item.get("text")'
                # dengan nama field yang sebenarnya di file JSON Anda.
                doc_name = str(item.get("id", f"{i+1}")) # Gunakan indeks jika ID tidak ada
                doc_content = item.get("text", "")
                if doc_content:
                    docs.append(doc_content)
                    doc_names.append(doc_name)
        if not docs:
            st.sidebar.warning("Tidak ada dokumen yang ditemukan dalam file JSON atau format tidak sesuai.")
    except FileNotFoundError:
        st.sidebar.error(f"File JSON tidak ditemukan di: {file_path}")
        return [], []
    except json.JSONDecodeError:
        st.sidebar.error(f"Gagal membaca file JSON. Pastikan formatnya benar: {file_path}")
        return [], []
    except Exception as e:
        st.sidebar.error(f"Terjadi kesalahan saat memuat JSON: {e}")
        return [], []
    return docs, doc_names

# --- UI Streamlit ---
st.set_page_config(layout="wide") # Opsi: Menggunakan layout lebar untuk tabel

st.markdown("<h1 style='text-align: center;'>Mini Search Engine</h1>", unsafe_allow_html=True)
# st.title("Sistem Temu Kembali Informasi")

# Sidebar untuk Kontrol
st.sidebar.header("Pengaturan Data Dokumen")

st.sidebar.write(f"Dataset: `{os.path.join(DATA_SUBFOLDER, DATASET_FILENAME)}`")
if st.sidebar.button("Muat Dokumen"):
    if os.path.exists(STATIC_JSON_FILE_PATH):
        with st.spinner("Memuat dan memproses dokumen dari JSON..."):
            st.session_state.raw_docs, st.session_state.doc_names = load_documents_from_json(STATIC_JSON_FILE_PATH)
            if st.session_state.raw_docs:
                st.sidebar.success(f"{len(st.session_state.raw_docs)} dokumen berhasil dimuat!")
                st.session_state.engine = None
                st.session_state.predicted_ids_for_eval = []
                st.session_state.current_engine_key = None
            # Pesan error/warning akan ditampilkan oleh load_documents_from_json
    else:
        st.sidebar.error(f"File dataset '{DATASET_FILENAME}' tidak ditemukan di folder '{DATA_SUBFOLDER}'.")
        st.sidebar.info(f"Path yang dicari: {STATIC_JSON_FILE_PATH}")

if st.session_state.raw_docs:
    st.sidebar.header("Pengaturan Model Pencarian")
    st.sidebar.write(f"Jumlah dokumen: {len(st.session_state.raw_docs)}")
    
    model_choice = st.sidebar.selectbox("Pilih Model:", ["VSM", "BM25"])
    k_param, b_param = 1.5, 0.75
    if model_choice == "BM25":

        k_param = st.sidebar.slider("Parameter k (BM25):", 0.0, 3.0, 1.5, 0.1) # min, max, place holder, kenaikan tiap
        b_param = st.sidebar.slider("Parameter b (BM25):", 0.0, 1.0, 0.75, 0.05) # min, max, place holder, kenaikan tiap

    engine_key = f"{model_choice}_{k_param}_{b_param}" if model_choice == "BM25" else model_choice
    
    if st.session_state.current_engine_key != engine_key or st.session_state.engine is None:
        with st.spinner(f"Menginisialisasi engine {model_choice}..."):
            st.session_state.engine = load_engine(model_choice, st.session_state.raw_docs, k_param, b_param)
            st.session_state.current_engine_key = engine_key
            if st.session_state.engine:
                st.sidebar.success(f"Engine {model_choice} siap!")
            else:
                st.sidebar.error(f"Gagal menginisialisasi engine {model_choice}.")

    top_k_max_val = len(st.session_state.raw_docs)
    top_k_default_val = min(5, top_k_max_val) if top_k_max_val > 0 else 1
    top_k = st.sidebar.number_input("Jumlah hasil teratas (Top-K):", min_value=1, max_value=top_k_max_val if top_k_max_val > 0 else 1, value=top_k_default_val)

    st.subheader("Masukkan Kueri Pencarian")
    query = st.text_input("Kueri:")

    if st.button("Cari Dokumen"):
        if not query:
            st.warning("Harap masukkan kueri.")
        elif not st.session_state.engine:
            st.error("Engine belum siap. Silakan muat dokumen terlebih dahulu.")
        else:
            with st.spinner("Mencari..."):
                ranked_ids, scores = st.session_state.engine.search(query, top_k=top_k) #

            st.subheader(f"Hasil Pencarian untuk: '{query}'")
            if not ranked_ids:
                st.write("Tidak ada dokumen yang relevan ditemukan.")
            else:
                results_data = []
                for rank, doc_id in enumerate(ranked_ids):
                    doc_name = st.session_state.doc_names[doc_id] if doc_id < len(st.session_state.doc_names) else f"Dokumen ID {doc_id}"
                    score_value = scores[doc_id] if doc_id < len(scores) else "N/A"
                    preview_text = st.session_state.raw_docs[doc_id][:250] + "..." if doc_id < len(st.session_state.raw_docs) else "N/A"
                    results_data.append({
                        "Peringkat": rank + 1,
                        "Doc_ID": doc_name,
                        "Skor": score_value, # Format akan diatur di column_config
                        "Preview": preview_text
                    })
                
                if results_data:
                    df_to_display = pd.DataFrame(results_data)
                    st.dataframe(
                        df_to_display,
                        column_config={
                            "Peringkat": st.column_config.NumberColumn(width="small"),
                            "Nama Dokumen/ID": st.column_config.TextColumn(width="medium"),
                            "Skor": st.column_config.NumberColumn(width="small", format="%.4f"),
                            "Preview": st.column_config.TextColumn(width="large")
                        },
                        hide_index=True, # Menyembunyikan indeks 0,1,2.. dari DataFrame
                        use_container_width=True # Opsi: Agar tabel menggunakan lebar penuh kontainer
                    )
                    st.session_state.predicted_ids_for_eval = ranked_ids
                else:
                    st.write("Tidak ada data hasil untuk ditampilkan.")


    if st.session_state.predicted_ids_for_eval:
        st.markdown("---")
        st.subheader("Evaluasi Hasil")
        relevant_docs_str = st.text_input(
            "Masukkan ID Dokumen Relevan (pisahkan dengan koma, contoh: 0,2,5):",
            help="Gunakan indeks dokumen (mulai dari 0) sesuai urutan pemuatan awal."
        )

        if st.button("Hitung Metrik Evaluasi"):
            if relevant_docs_str:
                try:
                    relevant_ids = [int(id_str.strip()) for id_str in relevant_docs_str.split(',')]
                    predicted_ids = st.session_state.predicted_ids_for_eval
                    
                    precision, recall, f1 = precision_recall_f1(predicted_ids, relevant_ids) #
                    
                    st.metric(label="Precision", value=f"{precision:.4f}")
                    st.metric(label="Recall", value=f"{recall:.4f}")
                    st.metric(label="F1-Score", value=f"{f1:.4f}")
                except ValueError:
                    st.error("Format ID dokumen relevan salah. Harap masukkan angka yang dipisahkan koma.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat evaluasi: {e}")
            else:
                st.warning("Harap masukkan ID dokumen yang relevan untuk evaluasi.")
else:
    # st.info(f"Klik tombol 'Muat Dokumen' di sidebar untuk memuat dataset '{DATASET_FILENAME}'.")
    st.info(f"Untuk memulai pencarian, silahkan muat dokumen di Side Bar.")