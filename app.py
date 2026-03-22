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
        return joblib.load('model_dropout.pkl')
    except Exception as e:
        st.error(f"Error Loading Model: {e}. Pastikan versi scikit-learn adalah 1.5.1")
        return None

@st.cache_data
def load_data():
    return pd.read_csv("data.csv", sep=";")

model = load_model()
df_raw = load_data()

# 3. Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["📊 Dashboard Analisis", "🔍 Prediksi Kelulusan"])

# --- HALAMAN 1: DASHBOARD ---
if page == "📊 Dashboard Analisis":
    st.title("📊 Dashboard Strategis Mahasiswa")
    
    # FILTER INTERAKTIF
    st.sidebar.divider()
    st.sidebar.subheader("Filter Data")
    f_gender = st.sidebar.multiselect("Gender:", options=df_raw['Gender'].unique(), default=df_raw['Gender'].unique(), format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
    f_scholar = st.sidebar.multiselect("Beasiswa:", options=df_raw['Scholarship_holder'].unique(), default=df_raw['Scholarship_holder'].unique(), format_func=lambda x: "Ya" if x==1 else "Tidak")

    df_filtered = df_raw[(df_raw['Gender'].isin(f_gender)) & (df_raw['Scholarship_holder'].isin(f_scholar))]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Usia & Status")
        fig1 = px.histogram(df_filtered, x="Age_at_enrollment", color="Status", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("Nilai Semester 2 vs Status")
        fig2 = px.box(df_filtered, x="Status", y="Curricular_units_2nd_sem_grade", color="Status")
        st.plotly_chart(fig2, use_container_width=True)

# --- HALAMAN 2: PREDIKSI ---
else:
    st.title("🔍 Prediksi Potensi Dropout")
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
                tuition = st.selectbox("Biaya Kuliah Lunas?", options=[(1, "Ya"), (0, "Tidak")])[0]
                scholarship = st.selectbox("Penerima Beasiswa?", options=[(1, "Ya"), (0, "Tidak")])[0]
                sem2_grade = st.number_input("Nilai Rata-rata Sem 2", value=12.0)
            with c2:
                debtor = st.selectbox("Memiliki Hutang?", options=[(1, "Ya"), (0, "Tidak")])[0]
                age = st.number_input("Usia Saat Daftar", value=20)
                gender = st.selectbox("Gender", options=[(1, "Laki-laki"), (0, "Perempuan")])[0]
            
            submit = st.form_submit_button("Prediksi")

        if submit:
            data_input = {feat: 0 for feat in expected_features}
            data_input.update({'Tuition_fees_up_to_date': tuition, 'Scholarship_holder': scholarship, 'Curricular_units_2nd_sem_grade': sem2_grade, 'Debtor': debtor, 'Age_at_enrollment': age, 'Gender': gender})
            df_final = pd.DataFrame([data_input])[expected_features]
            res = model.predict(df_final)[0]
            
            if res == 0: st.success("### HASIL: GRADUATE ✅")
            else: st.error("### HASIL: DROPOUT ❌")