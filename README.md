# 💸 SPENDIGO 
 Analisis Keuangan Gen-Z | Coding Camp 2026

## 🔗 Live Dashboard
(https://spendigo.streamlit.app)

## 📁 Struktur Repo
ANALISIS-DATA-SPENDIGO/
├── dashboard/
│   ├── app.py                        # Streamlit dashboard utama
│   ├── requirements.txt              # Dependensi Python
│   ├── genz_financial_dataset_v3.csv # Dataset hasil feature engineering
│   ├── rf_model.pkl                  # Model Random Forest terlatih
│   └── le_user_type.pkl              # Label Encoder User Type
│
├── notebook/
│   ├── CapstoneSpendigo.ipynb        # Notebook analisis lengkap
│   └── genz_financial_dataset_v3.csv # Dataset
│
└── .gitattributes
└── README.md

## 🛠️ Tech Stack
Python · Pandas · NumPy · Scikit-learn · TensorFlow · SciPy · Matplotlib · Seaborn · Streamlit

## ⚙️ Cara Menjalankan Lokal
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
