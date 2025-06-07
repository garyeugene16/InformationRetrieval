@echo off
REM 1. Cek dulu apakah folder venv sudah ada
IF NOT EXIST venv (
    echo Membuat virtual environment...
    python -m venv venv
)

REM 2. Aktifin virtual environment
call venv\Scripts\activate

REM 3. Install dependencies jika requirements.txt ada
IF EXIST requirements.txt (
    echo Menginstall dependencies...
    pip install -r requirements.txt
)

REM 4. Jalankan aplikasi Streamlit
echo Menjalankan aplikasi...
streamlit run GUIsearch.py

pause