import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL & FITUR
# =========================
model = joblib.load("rf_prediksi_covid_new_cases.pkl")
fitur_prediktor = joblib.load("fitur_prediktor_covid.pkl")

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Kasus Baru COVID-19",
    page_icon="🦠",
    layout="centered"
)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <h2 style='text-align:center;'>Prediksi Kasus Baru COVID-19</h2>
    <p style='text-align:center; color:gray;'>
    Berdasarkan Data Demografi dan Kondisi Epidemiologis Wilayah
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =========================
# FORM INPUT USER
# =========================
with st.form("form_prediksi"):
    st.subheader("Data Wilayah")

    population = st.number_input(
        "Jumlah Penduduk",
        min_value=0,
        value=1000000,
        step=1000
    )

    pop_density = st.number_input(
        "Kepadatan Penduduk (jiwa/km²)",
        min_value=0.0,
        value=1000.0
    )

    total_cases = st.number_input(
        "Total Kasus Kumulatif",
        min_value=0,
        value=50000
    )

    total_deaths = st.number_input(
        "Total Kematian",
        min_value=0,
        value=1000
    )

    total_recovered = st.number_input(
        "Total Sembuh",
        min_value=0,
        value=45000
    )

    active_cases = st.number_input(
        "Total Kasus Aktif",
        min_value=0,
        value=4000
    )

    area = st.number_input(
        "Luas Wilayah (km²)",
        min_value=0.0,
        value=1500.0
    )

    regencies = st.number_input(
        "Jumlah Kabupaten",
        min_value=0,
        value=10
    )

    cities = st.number_input(
        "Jumlah Kota",
        min_value=0,
        value=5
    )

    districts = st.number_input(
        "Jumlah Kecamatan",
        min_value=0,
        value=120
    )

    submit = st.form_submit_button("Prediksi Kasus Baru")

# =========================
# PROSES PREDIKSI
# =========================
if submit:
    input_data = pd.DataFrame([[
        population,
        pop_density,
        total_cases,
        total_deaths,
        total_recovered,
        active_cases,
        area,
        regencies,
        cities,
        districts
    ]], columns=fitur_prediktor)

    # Prediksi
    prediksi = model.predict(input_data)[0]
    prediksi = int(round(prediksi))

    # =========================
    # OUTPUT HASIL
    # =========================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")

    st.markdown(
        f"""
        ### Perkiraan Kasus Baru COVID-19:
        <span style="font-weight:600;">{prediksi:,} kasus</span>
        """,
        unsafe_allow_html=True
    )

    # Interpretasi
    if prediksi < 100:
        st.success("Risiko penambahan kasus tergolong rendah.")
    elif prediksi < 500:
        st.info("Terjadi penambahan kasus dalam tingkat sedang.")
    elif prediksi < 1000:
        st.warning("Potensi lonjakan kasus cukup tinggi.")
    else:
        st.error("Waspada! Potensi lonjakan kasus sangat tinggi.")
