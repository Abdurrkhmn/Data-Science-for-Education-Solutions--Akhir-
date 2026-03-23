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
        # Load model sesuai versi scikit-learn di requirements.txt (1.8.0)
        return joblib.load('model_dropout.pkl')
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv", sep=";")
        # Kriteria 3: Menghapus 'Enrolled' agar konsisten dengan notebook (Biner: Graduate vs Dropout)
        if 'Status' in df.columns:
            df = df[df['Status'] != 'Enrolled']
        
        # Bersihkan spasi di nama kolom jika ada
        df.columns = df.columns.str.strip()

        # FIX AGAR GRAFIK MUNCUL: Konversi koma ke titik jika ada
        if 'Curricular_units_2nd_sem_grade' in df.columns:
            df['Curricular_units_2nd_sem_grade'] = pd.to_numeric(df['Curricular_units_2nd_sem_grade'].astype(str).str.replace(',', '.'), errors='coerce')

        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

model = load_model()
df_raw = load_data()

# --- 3. SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["📊 Dashboard Analisis", "🔍 Prediksi Kelulusan"])

# --- 4. HALAMAN 1: DASHBOARD ---
if page == "📊 Dashboard Analisis":
    st.title("📊 Dashboard Strategis Mahasiswa Jaya Jaya Institut")
    st.markdown("Analisis komprehensif faktor penentu keberhasilan studi mahasiswa.")
    
    st.sidebar.divider()
    st.sidebar.subheader("Filter Data")
    
    gender_map = {1: "Laki-laki", 0: "Perempuan"}
    scholar_map = {1: "Penerima Beasiswa", 0: "Bukan Penerima"}
    
    selected_gender_labels = st.sidebar.multiselect(
        "Gender:", 
        options=list(gender_map.values()), 
        default=list(gender_map.values())
    )
    
    selected_scholar_labels = st.sidebar.multiselect(
        "Status Beasiswa:", 
        options=list(scholar_map.values()), 
        default=list(scholar_map.values())
    )

    inv_gender_map = {v: k for k, v in gender_map.items()}
    inv_scholar_map = {v: k for k, v in scholar_map.items()}
    
    genders = [inv_gender_map[l] for l in selected_gender_labels]
    scholars = [inv_scholar_map[l] for l in selected_scholar_labels]

    df_filtered = df_raw[
        (df_raw['Gender'].isin(genders)) & 
        (df_raw['Scholarship_holder'].isin(scholars))
    ]

    if df_filtered.empty:
        st.warning("⚠️ Data tidak ditemukan untuk kombinasi filter ini.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Mahasiswa", len(df_filtered))
        with m2:
            drp = len(df_filtered[df_filtered['Status'] == 'Dropout'])
            pct = (drp/len(df_filtered)*100) if len(df_filtered) > 0 else 0
            st.metric("Total Dropout", drp, f"{pct:.1f}%", delta_color="inverse")
        with m3:
            avg_g = df_filtered['Curricular_units_2nd_sem_grade'].mean()
            st.metric("Rata-rata Nilai Sem 2", f"{avg_g:.2f}")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📌 Komposisi Status Mahasiswa")
            fig_pie = px.pie(df_filtered, names='Status', hole=0.4,
                             color='Status',
                             color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.subheader("📌 Status Berdasarkan Kelompok Usia")
            df_filtered['Age_Group'] = pd.cut(df_filtered['Age_at_enrollment'], 
                                            bins=[0, 20, 30, 40, 60], 
                                            labels=['<20', '20-30', '30-40', '>40'])
            fig_age = px.histogram(df_filtered, x='Age_Group', color='Status', barmode='group',
                                   color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
            st.plotly_chart(fig_age, use_container_width=True)

        st.divider()

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("📌 Pengaruh Hutang Biaya Kuliah")
            df_filtered['Hutang'] = df_filtered['Debtor'].map({1: 'Ada Hutang', 0: 'Lunas'})
            fig_debt = px.histogram(df_filtered, x='Hutang', color='Status', barmode='stack',
                                    color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
            st.plotly_chart(fig_debt, use_container_width=True)

        with c4:
            st.subheader("📌 Distribusi Nilai Semester 2")
            fig_hist = px.histogram(df_filtered, x="Curricular_units_2nd_sem_grade", color="Status",
                                    marginal="box", color_discrete_map={'Dropout': '#ef553b', 'Graduate': '#636efa'})
            st.plotly_chart(fig_hist, use_container_width=True)

# --- 5. HALAMAN 2: PREDIKSI ---
else:
    st.title("🔍 Prediksi Potensi Kelulusan")
    st.info("Sistem ini menggunakan 36 fitur akademik dan demografis sesuai dengan model latih.")

    if model is not None:
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

        with st.form("input_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Data Personal & Ekonomi")
                f_gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
                f_scholar = st.selectbox("Beasiswa", [1, 0], format_func=lambda x: "Penerima" if x==1 else "Tidak")
                f_debtor = st.selectbox("Hutang Kuliah", [1, 0], format_func=lambda x: "Ada" if x==1 else "Tidak Ada")
                f_tuition = st.selectbox("UKT Lunas?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")
                f_age = st.number_input("Usia Saat Daftar", 17, 60, 20)
            
            with col_b:
                st.subheader("Data Akademik")
                f_course = st.number_input("ID Program Studi (Course)", 1, 9999, 33)
                f_admission = st.number_input("Nilai Masuk (Admission Grade)", 0.0, 200.0, 120.0)
                f_sem2_approved = st.number_input("Unit Sem 2 Lulus", 0, 30, 5)
                f_sem2_grade = st.number_input("Nilai Rata-rata Sem 2", 0.0, 20.0, 12.0)
                f_displaced = st.selectbox("Mahasiswa Perantau?", [1, 0], format_func=lambda x: "Ya" if x==1 else "Tidak")

            st.divider()
            submit = st.form_submit_button("Mulai Analisis Prediksi")

        if submit:
            input_data = {feat: 0 for feat in features}
            input_data.update({
                'Gender': f_gender, 'Scholarship_holder': f_scholar, 'Debtor': f_debtor,
                'Tuition_fees_up_to_date': f_tuition, 'Age_at_enrollment': f_age,
                'Course': f_course, 'Admission_grade': f_admission,
                'Curricular_units_2nd_sem_approved': f_sem2_approved,
                'Curricular_units_2nd_sem_grade': f_sem2_grade,
                'Displaced': f_displaced,
                'Curricular_units_1st_sem_approved': f_sem2_approved,
                'Curricular_units_1st_sem_grade': f_sem2_grade,
                'Curricular_units_1st_sem_enrolled': f_sem2_approved + 1,
                'Curricular_units_2nd_sem_enrolled': f_sem2_approved + 1
            })

            df_pred = pd.DataFrame([input_data])[features]
            res = model.predict(df_pred)[0]

            st.subheader("Hasil Analisis:")
            if res == 0 or str(res).lower() == 'graduate':
                st.success("### STATUS PREDIKSI: GRADUATE (LULUS) ✅")
                st.balloons()
                st.markdown("""
                **Saran Strategis:**
                * **Career Preparation:** Mahasiswa berada di jalur yang benar. Sarankan untuk mulai mengambil sertifikasi profesional.
                * **Ambassador:** Mahasiswa ini berpotensi menjadi mentor bagi adik tingkat.
                """)
            else:
                st.error("### STATUS PREDIKSI: DROPOUT (BERISIKO) ❌")
                st.markdown("""
                **Saran Intervensi (Action Items):**
                * **Early Warning:** Segera jadwalkan pertemuan dengan Dosen Pembimbing Akademik.
                * **Financial Check:** Jika faktor penyebab adalah hutang/biaya, arahkan ke bagian kemahasiswaan untuk cicilan.
                * **Tutoring:** Berikan tambahan jam belajar untuk mata kuliah semester 2 yang sulit.
                """)