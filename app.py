import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Jaya Jaya Institut Analytics", layout="wide")

# --- 2. LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_dropout.pkl')
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        # Load data dengan pemisah titik koma
        df = pd.read_csv("data.csv", sep=";")
        df.columns = df.columns.str.strip()
        
        # Hapus status 'Enrolled' agar konsisten dengan biner (Graduate/Dropout)
        if 'Status' in df.columns:
            df = df[df['Status'] != 'Enrolled']

        # SOLUSI VISUALISASI KOSONG: Paksa kolom nilai menjadi tipe angka (float)
        cols_to_fix = ['Curricular_units_2nd_sem_grade', 'Age_at_enrollment', 'Admission_grade']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Hapus data yang gagal dikonversi (NaN) agar tidak merusak grafik
        df = df.dropna(subset=['Curricular_units_2nd_sem_grade'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

model = load_model()
df_raw = load_data()

# --- 3. SIDEBAR NAVIGASI & FILTER ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["📊 Dashboard Analisis", "🔍 Prediksi Kelulusan"])

# FILTER GLOBAL (Sesuai Laporan: Gender & Beasiswa)
st.sidebar.divider()
st.sidebar.subheader("Filter Dashboard")

gender_options = {1: "Laki-laki", 0: "Perempuan"}
scholar_options = {1: "Penerima Beasiswa", 0: "Bukan Penerima"}

sel_gender = st.sidebar.multiselect("Gender:", options=[1, 0], default=[1, 0], format_func=lambda x: gender_options[x])
sel_scholar = st.sidebar.multiselect("Status Beasiswa:", options=[1, 0], default=[1, 0], format_func=lambda x: scholar_options[x])

# Terapkan Filter
df_filtered = df_raw[
    (df_raw['Gender'].isin(sel_gender)) & 
    (df_raw['Scholarship_holder'].isin(sel_scholar))
].copy()

# --- 4. HALAMAN 1: DASHBOARD ---
if page == "📊 Dashboard Analisis":
    st.title("📊 Dashboard Strategis Mahasiswa")
    
    if df_filtered.empty:
        st.warning("Data kosong. Silakan sesuaikan filter di sidebar.")
    else:
        # Baris 1: Metrik
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Total Mahasiswa", len(df_filtered))
        with m2: 
            drp = len(df_filtered[df_filtered['Status'] == 'Dropout'])
            st.metric("Total Dropout", drp, f"{(drp/len(df_filtered)*100):.1f}%", delta_color="inverse")
        with m3: st.metric("Rata-rata Nilai Sem 2", f"{df_filtered['Curricular_units_2nd_sem_grade'].mean():.2f}")

        st.divider()

        # Baris 2: Grafik
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📌 Komposisi Status")
            fig1 = px.pie(df_filtered, names='Status', hole=0.4, color='Status',
                         color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.subheader("📌 Distribusi Nilai Semester 2")
            fig2 = px.histogram(df_filtered, x="Curricular_units_2nd_sem_grade", color="Status",
                                barmode="overlay", marginal="box",
                                color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
            st.plotly_chart(fig2, use_container_width=True)

        # Baris 3: Tambahan Insight
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("📌 Status Berdasarkan Kelompok Usia")
            df_filtered['Age_Group'] = pd.cut(df_filtered['Age_at_enrollment'], bins=[0, 25, 40, 60], labels=['Muda', 'Dewasa', 'Tua'])
            fig3 = px.histogram(df_filtered, x='Age_Group', color='Status', barmode='group',
                               color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            st.subheader("📌 Pengaruh Hutang Kuliah")
            df_filtered['Hutang'] = df_filtered['Debtor'].map({1: 'Ada', 0: 'Tidak'})
            fig4 = px.histogram(df_filtered, x='Hutang', color='Status', barmode='group',
                               color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
            st.plotly_chart(fig4, use_container_width=True)

# --- 5. HALAMAN 2: PREDIKSI ---
else:
    st.title("🔍 Prediksi Potensi Kelulusan")
    # Sesuai fitur model (36 fitur)
    if model:
        with st.form("predict_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                f_gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
                f_scholar = st.selectbox("Penerima Beasiswa?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
                f_debtor = st.selectbox("Memiliki Hutang?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
                f_age = st.number_input("Usia Saat Daftar", 17, 60, 20)
            with col_b:
                f_admission = st.number_input("Nilai Masuk", 0.0, 200.0, 120.0)
                f_sem2_grade = st.number_input("Nilai Sem 2", 0.0, 20.0, 12.0)
                f_sem2_approved = st.number_input("Unit Sem 2 Lulus", 0, 30, 5)
                f_tuition = st.selectbox("UKT Lunas?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")

            if st.form_submit_button("Cek Status"):
                # Buat DataFrame dengan 36 kolom (default 0)
                features = model.feature_names_in_
                input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
                
                # Isi fitur yang relevan
                input_data['Gender'] = f_gender
                input_data['Scholarship_holder'] = f_scholar
                input_data['Debtor'] = f_debtor
                input_data['Age_at_enrollment'] = f_age
                input_data['Admission_grade'] = f_admission
                input_data['Curricular_units_2nd_sem_grade'] = f_sem2_grade
                input_data['Curricular_units_2nd_sem_approved'] = f_sem2_approved
                input_data['Tuition_fees_up_to_date'] = f_tuition
                
                res = model.predict(input_data)[0]
                if res == 0 or str(res).lower() == 'graduate':
                    st.success("### HASIL: GRADUATE ✅")
                else:
                    st.error("### HASIL: DROPOUT ❌")