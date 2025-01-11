import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Judul Aplikasi
st.title("Aplikasi Keputusan Etis")

# Deskripsi Aplikasi
st.write("""
Aplikasi ini membantu Anda membuat keputusan etis berdasarkan input yang Anda berikan.
Silakan masukkan informasi tentang keputusan yang akan diambil, dan model akan memberikan rekomendasi.
""")

# Input dari pengguna
education = st.selectbox('Pendidikan', ['Bachelor', 'Master', 'PhD'])
job_type = st.selectbox('Jenis Pekerjaan', ['Private', 'Public', 'Self-Employed'])
age = st.number_input('Usia', min_value=18, max_value=100, value=25)

# Mempersiapkan data untuk prediksi
data = pd.DataFrame({
    'education': [education],
    'job_type': [job_type],
    'age': [age]
})

# Fungsi untuk mempersiapkan dan melatih model
@st.cache_data
def load_model():
    # Membaca dataset dan mempersiapkan data
    try:
        df = pd.read_csv('data.csv')  # Gantilah dengan path dataset Anda
    except FileNotFoundError:
        st.error("File data.csv tidak ditemukan. Pastikan file CSV ada di direktori yang benar.")
        return None, None, None

    # Menampilkan kolom dataset untuk memverifikasi
    st.write("Kolom dalam dataset:", df.columns)

    label_encoder = LabelEncoder()
    
    # Cek apakah kolom 'education' dan 'ethical_decision' ada dalam DataFrame
    if 'education' not in df.columns or 'ethical_decision' not in df.columns:
        st.error("Kolom 'education' atau 'ethical_decision' tidak ditemukan dalam dataset.")
        return None, None, None

    # Melakukan encoding pada kolom 'education' dan 'ethical_decision'
    df['education'] = label_encoder.fit_transform(df['education'])
    df['ethical_decision'] = label_encoder.fit_transform(df['ethical_decision'])
    
    X = df.drop('ethical_decision', axis=1)  # 'ethical_decision' adalah kolom target Anda
    y = df['ethical_decision']
    
    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat dan melatih model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model, label_encoder, df

# Memanggil model dan label_encoder
model, label_encoder, df = load_model()

# Melakukan prediksi jika model berhasil dimuat
if model and label_encoder and df is not None:
    # Menggabungkan data input pengguna dengan data pelatihan untuk menangani label baru
    df_combined = pd.concat([df[['education']], data[['education']]], ignore_index=True)
    
    # Ensure the 'education' column contains only strings
    df_combined['education'] = df_combined['education'].astype(str)
    
    # Melakukan fit_transform pada gabungan data (training dan input pengguna)
    label_encoder.fit(df_combined['education'])
    
    # Transformasi input data pengguna
    data['education'] = label_encoder.transform(data['education'].astype(str))
    
    # Ensure the input data has the same feature names as the training data
    data = data.reindex(columns=df.columns.drop('ethical_decision'), fill_value=0)
    
    # Prediksi keputusan
    prediction = model.predict(data)
    
    # Menampilkan hasil prediksi
    if prediction == 1:
        st.write("**Rekomendasi:** Keputusan ini bisa dianggap etis.")
    else:
        st.write("**Rekomendasi:** Keputusan ini mungkin tidak etis.")
