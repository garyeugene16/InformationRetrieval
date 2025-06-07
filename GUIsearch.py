import os
import json
import streamlit as st # type: ignore
import pandas as pd
from search_engine.preprocessing import preprocess
from search_engine.vsm import VSMEngine
from search_engine.bm25 import BM25Engine

# Modifikasi st.session_state agar sinkron dengan search2.py
if 'raw_docs' not in st.session_state:
    st.session_state.raw_docs = []
    st.session_state.doc_names = []
    st.session_state.doc_ids = [] 
    st.session_state.doc_lookup = {}  
    st.session_state.engine = None
    st.session_state.current_engine_key = None
    st.session_state.current_file = None

# --- Cache untuk performa ---
@st.cache_resource
def load_engine(engine_type, docs, k=1.5, b=0.75):
    if engine_type == "BM25":
        return BM25Engine(docs, k, b)
    elif engine_type == "VSM":
        return VSMEngine(docs)
    return None

# --- FUngsi untuk fetch file di data ---
def get_document_files():
    """Mendapatkan daftar file dokumen yang valid"""
    return [f for f in os.listdir('data') if f.startswith('documents') and f.endswith('.json')]

# --- Fungsi yang copy dari search2.py ---
def load_documents_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
            
        raw_texts = [doc["text"] for doc in docs]
        doc_ids = [doc["doc_id"] for doc in docs]
        doc_lookup = {doc["doc_id"]: doc["text"] for doc in docs}
        
        if not raw_texts:
            st.sidebar.warning("Tidak ada dokumen yang ditemukan dalam file JSON atau format tidak sesuai.")
            return [], [], [], {}
            
        return raw_texts, doc_ids, doc_ids, doc_lookup  
        
    except FileNotFoundError:
        st.sidebar.error(f"File JSON tidak ditemukan di: {file_path}")
        return [], [], [], {}
    except json.JSONDecodeError:
        st.sidebar.error(f"Gagal membaca file JSON. Pastikan formatnya benar: {file_path}")
        return [], [], [], {}
    except Exception as e:
        st.sidebar.error(f"Terjadi kesalahan saat memuat JSON: {e}")
        return [], [], [], {}

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


# --- UI Streamlit ---
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>Mini Search Engine</h1>", unsafe_allow_html=True)

# Sidebar untuk atur data dan model
st.sidebar.header("Pengaturan Data Dokumen")

# Untuk footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: left;
        font-size: 15px;
        color: gray;
        padding: 10px 0px;
    }
    </style>

    <div class="footer">
        Tugas Information Retrieval oleh : Rio Sugiarno (6182201043), Gary Eugene (6182201046), Hepi Rahmat Stevanus Daeli (6182201052), dan Albert Christian Lifen (6182201055)
    </div>
    """,
    unsafe_allow_html=True
)

# Pilihan file dokumen
document_files = get_document_files()
if not document_files:
    st.sidebar.error("Tidak ada file dokumen yang valid di folder 'data'.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Pilih File Dokumen:",
    document_files,
    index=0
)
# Proses dokumen pilihan pengguna
if st.sidebar.button("Muat Dokumen"):
    file_path = os.path.join('data', selected_file)
    with st.spinner("Memuat dan memproses dokumen dari JSON..."):
        st.session_state.raw_docs, st.session_state.doc_ids, st.session_state.doc_names, st.session_state.doc_lookup = load_documents_from_json(file_path)
        if st.session_state.raw_docs:
            st.sidebar.success(f"{len(st.session_state.raw_docs)} dokumen berhasil dimuat!")
            st.session_state.engine = None
            st.session_state.current_engine_key = None
            st.session_state.current_file = selected_file

if st.session_state.raw_docs:
    # Log banyak dokumen dan pilihan dokumen
    st.sidebar.header("Pengaturan Model Pencarian")
    st.sidebar.write(f"Jumlah dokumen: {len(st.session_state.raw_docs)}")
    st.sidebar.write(f"File aktif: {st.session_state.current_file}")

    # Buat pilihan model
    model_choice = st.sidebar.selectbox("Pilih Model:", ["VSM", "BM25"])
    k_param = 1.5
    b_param = 0.75

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
    top_k = st.sidebar.number_input("Jumlah hasil teratas (Top-K):", 
                                  min_value=1, 
                                  max_value=top_k_max_val if top_k_max_val > 0 else 1, 
                                  value=top_k_default_val)

    st.subheader("Masukkan Kueri Pencarian")
    query = st.text_input("Kueri:")

    if st.button("Cari Dokumen"):
        if not query:
            st.warning("Harap masukkan kueri.")
        elif not st.session_state.engine:
            st.error("Engine belum siap. Silakan muat dokumen terlebih dahulu.")
        else:
            with st.spinner("Mencari..."):
                ranked_ids, scores = st.session_state.engine.search(query, top_k=top_k)
                    # sorted_scores = sorted(scores, reverse=True)
                    # print(sorted_scores)
            st.subheader(f"Hasil Pencarian untuk: '{query}'")
            if not ranked_ids or all(scores[i] == 0 for i in ranked_ids):
                st.write("Tidak ada dokumen yang relevan ditemukan.")
            else:
                results_data = []
                for rank, doc_id in enumerate(ranked_ids):
                    # Gunakan doc_ids seperti di search2.py
                    doc_name = st.session_state.doc_ids[doc_id]
                    score_value = scores[doc_id]
                    full_text = st.session_state.doc_lookup[doc_name]
                    preview_text = generate_snippet(query, full_text)
                    results_data.append({
                        "Peringkat": rank + 1,
                        "Doc_ID": doc_name,
                        "Skor": score_value,
                        "Preview": preview_text
                    })
                
                if results_data:
                    df_to_display = pd.DataFrame(results_data)
                    st.dataframe(
                        df_to_display,
                        column_config={
                            "Peringkat": st.column_config.NumberColumn(width="small"),
                            "Doc_ID": st.column_config.TextColumn(width="medium"),
                            "Skor": st.column_config.NumberColumn(width="small", format="%.4f"),
                            "Preview": st.column_config.TextColumn(width="large")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
else:
    st.info("Untuk memulai pencarian, silahkan pilih dan muat dokumen di Side Bar.")