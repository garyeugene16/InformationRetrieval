#!/bin/bash
# Kalau bisa jalan karena permasalahan keamanan, gunakan createEnvAndRunGUI.sh

cd "$(dirname "$0")"

# 1. Cek apakah folder venv sudah ada
if [ ! -d "venv" ]; then
    echo "Membuat virtual environment..."
    python3 -m venv venv
fi

# 2. Aktifkan virtual environment
source venv/bin/activate

# 3. Install dependencies jika requirements.txt ada
if [ -f "requirements.txt" ]; then
    echo "Menginstall dependencies..."
    pip install -r requirements.txt
fi

# 4. Jalankan aplikasi Streamlit
echo "Menjalankan aplikasi..."
streamlit run GUIsearch.py

read -p "Tekan ENTER untuk menutup terminal..."
