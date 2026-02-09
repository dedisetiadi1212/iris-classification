import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Web Prediksi Bunga Iris", page_icon="🌸")

# --- LOAD MODEL ---
# Pastikan nama file sesuai dengan yang Anda simpan di Colab
model = joblib.load('model_iris_terbaik.sav')
iris_data = load_iris() # Untuk mengambil nama target bunga

# --- JUDUL APLIKASI ---
st.title("Aplikasi Klasifikasi Bunga Iris 🌸")
st.write("Silahkan masukkan ukuran kelopak dan mahkota untuk memprediksi jenis bunganya.")

# --- SIDEBAR UNTUK INPUT ---
st.sidebar.header("Input Parameter")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Menjalankan fungsi input
df_input = user_input_features()

# --- TAMPILAN UTAMA ---
st.subheader('Parameter yang Anda Masukkan:')
st.write(df_input)

# --- PROSES PREDIKSI ---
if st.button('Prediksi Sekarang!'):
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)

    st.subheader('Hasil Prediksi:')
    nama_bunga = iris_data.target_names[prediction][0]
    st.success(f"Model memprediksi ini adalah bunga: **{nama_bunga.capitalize()}**")

    # Menampilkan Probabilitas (Keyakinan Model)
    st.subheader('Probabilitas Prediksi:')
    proba_df = pd.DataFrame(prediction_proba, columns=iris_data.target_names)
    st.write(proba_df)

    st.divider()
    st.caption("Dibuat oleh **Dedi Setiadi**")