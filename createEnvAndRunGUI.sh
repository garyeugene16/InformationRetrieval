#!/bin/bash

# Note :
# 1. Buka terminal
# 2. Kasih izin jalankan dengan : chmod +x createEnvAndRunGUI.sh
# 3. Setelah dikasih izin, masukan command : ./createEnvAndRunGUI.sh

# 1. Cek dlo apakah folder venv sudah ada
if [ ! -d "venv" ]; then
    echo "Membuat virtual environment..."
    python3 -m venv venv
fi

# 2. Aktifin virtual environment
source venv/bin/activate

# 3. Install dependencies sesuai requirment . txt
if [ -f "requirements.txt" ]; then
    echo "Menginstall dependencies..."
    pip install -r requirements.txt
fi

# 4. Jalankan aplikasi Streamlit
echo "Menjalankan aplikasi..."
streamlit run GUIsearch.py
