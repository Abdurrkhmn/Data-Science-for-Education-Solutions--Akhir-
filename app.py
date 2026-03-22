import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Jaya Jaya Institut Analytics", layout="wide")

# 2. Load Model & Data
@st.cache_resource
def load_model():
    try:
        # Memastikan model dimuat dengan benar
        return joblib.load('model_dropout.pkl')
    except Exception as e:
        st.error(f"Error Loading Model: {e}. Pastikan versi scikit-learn di requirements.txt adalah 1.5.1")
        return None

@st.cache_data
def load_data():
    # Load data untuk dashboard
    return pd.read_csv("data.csv", sep=";")

model = load_model()
df_raw = load_data()

# 3. Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["📊 Dashboard Analisis", "🔍 Prediksi Kelulusan"])

# --- HALAMAN 1: DASHBOARD (Kriteria 3) ---
if page == "📊 Dashboard Analisis":
    st.title("📊 Dashboard Strategis Mahasiswa")
    st.markdown("Analisis faktor-faktor yang berkontribusi terhadap status mahasiswa.")
    
    # FILTER INTERAKTIF
    st.sidebar.divider()
    st.sidebar.subheader("Filter Dashboard")
    f_gender = st.sidebar.multiselect("Gender:", options=df_raw['Gender'].unique(), default=df_raw['Gender'].unique(), format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
    f_scholar = st.sidebar.multiselect("Penerima Beasiswa:", options=df_raw['Scholarship_holder'].unique(), default=df_raw['Scholarship_holder'].unique(), format_func=lambda x: "Ya" if x==1 else "Tidak")

    # Terapkan Filter
    df_filtered = df_raw[(df_raw['Gender'].isin(f_gender)) & (df_raw['Scholarship_holder'].isin(f_scholar))]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hubungan Usia vs Status")
        fig1 = px.histogram(df_filtered, x="Age_at_enrollment", color="Status", barmode="group",
                           labels={'Age_at_enrollment': 'Usia Saat Daftar', 'count': 'Jumlah Mahasiswa'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Nilai Semester 2 vs Status")
        fig2 = px.box(df_filtered, x="Status", y="Curricular_units_2nd_sem_grade", color="Status",
                     labels={'Curricular_units_2nd_sem_grade': 'Nilai Semester 2'})
        st.plotly_chart(fig2, use_container_width=True)

# --- HALAMAN 2: PREDIKSI ---
else:
    st.title("🔍 Prediksi Potensi Kelulusan")
    st.markdown("Masukkan data akademik mahasiswa untuk memprediksi status akhir.")
    
    if model is not None:
        expected_features = [
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
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Faktor Ekonomi & Dasar")
                tuition = st.selectbox("Biaya Kuliah Lunas?", options=[(1, "Ya"), (0, "Tidak")])[0]
                scholarship = st.selectbox("Penerima Beasiswa?", options=[(1, "Ya"), (0, "Tidak")])[0]
                debtor = st.selectbox("Memiliki Hutang?", options=[(1, "Ya"), (0, "Tidak")])[0]
                gender = st.selectbox("Gender", options=[(1, "Laki-laki"), (0, "Perempuan")])[0]
                
            with c2:
                st.subheader("Faktor Akademik")
                age = st.number_input("Usia Saat Daftar", min_value=17, max_value=60, value=20)
                sem2_grade = st.number_input("Prediksi Nilai Rata-rata Sem 2 (0-20)", min_value=0.0, max_value=20.0, value=12.0)
                admission = st.number_input("Nilai Masuk (Admission Grade)", value=120.0)
            
            submit = st.form_submit_button("Analisis Status Mahasiswa")

        if submit:
            # Buat dictionary awal semua fitur = 0
            data_input = {feat: 0 for feat in expected_features}
            
            # Update fitur dari input user
            # Penting: Curricular_units_2nd_sem_approved diisi otomatis berdasarkan grade 
            # agar model tidak menganggap mahasiswa tidak lulus unit sama sekali (penyebab dropout terus)
            approved_units = 5 if sem2_grade > 10 else 0
            
            data_input.update({
                'Tuition_fees_up_to_date': tuition,
                'Scholarship_holder': scholarship,
                'Curricular_units_2nd_sem_grade': sem2_grade,
                'Curricular_units_2nd_sem_approved': approved_units,
                'Debtor': debtor,
                'Age_at_enrollment': age,
                'Gender': gender,
                'Admission_grade': admission
            })

            # Buat DataFrame
            df_final = pd.DataFrame([data_input])[expected_features]
            
            # Prediksi (0 = Graduate, 1 = Dropout sesuai urutan label di notebook kamu)
            res = model.predict(df_final)[0]
            
            st.divider()
            if res == 0:
                st.success("### HASIL PREDIKSI: GRADUATE ✅")
                st.balloons()
                st.write("Mahasiswa diprediksi akan menyelesaikan studinya dengan baik.")
            else:
                st.error("### HASIL PREDIKSI: DROPOUT ❌")
                st.write("Mahasiswa berisiko tinggi putus kuliah. Disarankan intervensi akademik.")