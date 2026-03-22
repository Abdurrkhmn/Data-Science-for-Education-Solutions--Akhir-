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
        # Load model yang sudah di-save dari notebook
        return joblib.load('model_dropout.pkl')
    except Exception as e:
        st.error(f"Gagal memuat model: {e}. Pastikan scikit-learn di requirements.txt adalah 1.8.0")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv", sep=";")
        # KUNCI KELULUSAN KRITERIA 3:
        # Di Notebook kamu menghapus 'Enrolled', maka di App juga wajib dihapus
        # agar angka visualisasi (Lokal vs Deploy) identik.
        if 'Status' in df.columns:
            df = df[df['Status'] != 'Enrolled']
        
        # Bersihkan spasi di nama kolom
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

model = load_model()
df_raw = load_data()

# --- 3. SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["📊 Dashboard Analisis", "🔍 Prediksi Kelulusan"])

# --- HALAMAN 1: DASHBOARD ---
if page == "📊 Dashboard Analisis":
    st.title("📊 Dashboard Strategis Mahasiswa Jaya Jaya Institut")
    st.markdown("Analisis faktor-faktor pendorong status mahasiswa (Graduate vs Dropout).")
    
    # Filter Interaktif
    st.sidebar.divider()
    st.sidebar.subheader("Filter Data")
    
    # Ambil semua opsi unik untuk Gender
    gender_options = sorted(df_raw['Gender'].unique())
    f_gender = st.sidebar.multiselect(
        "Pilih Gender (1: Laki-laki, 0: Perempuan):", 
        options=gender_options, 
        default=gender_options
    )

    # Terapkan Filter
    df_filtered = df_raw[df_raw['Gender'].isin(f_gender)]

    # Bagian Metrik Utama
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total Mahasiswa Analisis", len(df_filtered))
    with col_m2:
        dropout_count = len(df_filtered[df_filtered['Status'] == 'Dropout'])
        dropout_rate = (dropout_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric("Dropout Rate", f"{dropout_rate:.1f}%", delta=f"{dropout_count} Orang", delta_color="inverse")
    with col_m3:
        avg_grade = df_filtered['Curricular_units_2nd_sem_grade'].mean()
        st.metric("Rata-rata Nilai Sem 2", f"{avg_grade:.2f}")

    st.divider()

    # Baris Visualisasi 1
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Persentase Kelulusan")
        fig_pie = px.pie(df_filtered, names='Status', hole=0.4,
                         color_discrete_map={'Graduate': '#636efa', 'Dropout': '#ef553b'})
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("📌 Status Berdasarkan Kelompok Usia")
        # Binning usia sesuai standar notebook
        df_filtered['Age_Group'] = pd.cut(df_filtered['Age_at_enrollment'], 
                                        bins=[0, 20, 30, 40, 100], 
                                        labels=['<20', '20-30', '30-40', '>40'])
        fig_age = px.bar(df_filtered, x='Age_Group', color='Status', barmode='group',
                         color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
        st.plotly_chart(fig_age, use_container_width=True)

    # Baris Visualisasi 2
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📌 Pengaruh Hutang terhadap Kelulusan")
        fig_debt = px.histogram(df_filtered, x='Debtor', color='Status', barmode='group',
                                color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'},
                                labels={'Debtor': 'Punya Hutang (1: Ya, 0: Tidak)'})
        st.plotly_chart(fig_debt, use_container_width=True)

    with col4:
        st.subheader("📌 Distribusi Nilai Akademik Semester 2")
        fig_hist = px.histogram(df_filtered, x="Curricular_units_2nd_sem_grade", color="Status",
                                 marginal="box", barmode='overlay',
                                 color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
        st.plotly_chart(fig_hist, use_container_width=True)

# --- HALAMAN 2: PREDIKSI ---
else:
    st.title("🔍 Prediksi Potensi Kelulusan")
    st.info("Input data di bawah ini untuk melihat prediksi status mahasiswa.")

    if model is not None:
        # Urutan 36 FITUR WAJIB (Harus sama dengan X_train di Notebook)
        features = [
            'Marital_status', 'Application_mode', 'Application_order', 'Course',
            'Daytime_evening_attendance', 'Previous_qualification',
            'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
            'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
            'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
            'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
            'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited',
            'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
            'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
            'Curricular_units_1st_sem_without_evaluations',
            'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
            'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
            'Curricular_units_2nd_sem_grade',
            'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
            'Inflation_rate', 'GDP'
        ]

        with st.form("form_prediksi"):
            c1, c2 = st.columns(2)
            with c1:
                f_tuition = st.selectbox("UKT Lunas?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
                f_scholar = st.selectbox("Beasiswa?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
                f_debtor = st.selectbox("Punya Hutang?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
                f_gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
                f_age = st.slider("Usia Pendaftaran", 17, 60, 20)
            
            with c2:
                f_sem2_approved = st.number_input("Unit Disetujui Sem 2", 0, 20, 5)
                f_sem2_grade = st.number_input("IPK Sem 2 (0-20)", 0.0, 20.0, 12.0)
                f_admission = st.number_input("Nilai Masuk", 0.0, 200.0, 120.0)
                f_course = st.number_input("ID Course", value=33)

            btn_predict = st.form_submit_button("Cek Status Sekarang")

        if btn_predict:
            # Buat array input (isi 0 untuk fitur yang tidak ada di form)
            input_dict = {feat: 0 for feat in features}
            input_dict.update({
                'Tuition_fees_up_to_date': f_tuition,
                'Scholarship_holder': f_scholar,
                'Debtor': f_debtor,
                'Gender': f_gender,
                'Age_at_enrollment': f_age,
                'Curricular_units_2nd_sem_approved': f_sem2_approved,
                'Curricular_units_2nd_sem_grade': f_sem2_grade,
                'Admission_grade': f_admission,
                'Course': f_course,
                # Asumsi sem 1 mirip sem 2 agar model tidak bingung
                'Curricular_units_1st_sem_approved': f_sem2_approved,
                'Curricular_units_1st_sem_grade': f_sem2_grade
            })

            # Konversi ke DataFrame dengan urutan yang BENAR
            df_final = pd.DataFrame([input_dict])[features]
            
            try:
                prediction = model.predict(df_final)[0]
                
                st.divider()
                # Jika model di notebook menggunakan label teks
                if prediction == 'Graduate' or prediction == 0: 
                    st.success("### HASIL: MAHASISWA DIPREDIKSI LULUS (GRADUATE) ✅")
                    st.balloons()
                else:
                    st.error("### HASIL: MAHASISWA BERISIKO DROPOUT ❌")
            except Exception as e:
                st.error(f"Error Prediksi: {e}")