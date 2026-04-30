import streamlit as st
import joblib
import pandas as pd

# cargar modelo
model = joblib.load("modelo_pipeline.pkl")

st.title("Predicción de precios de casas 🏠")

# cargar dataset original
df_original = pd.read_csv("train.csv")

# quitar target
df_original = df_original.drop(columns=["SalePrice"])

# valores por defecto
default_values = {}

for col in df_original.columns:
    if pd.api.types.is_numeric_dtype(df_original[col]):
        default_values[col] = df_original[col].median()
    else:
        mode = df_original[col].mode()
        default_values[col] = mode[0] if not mode.empty else "Unknown"

# inputs del usuario
overall_qual = st.slider("Calidad general (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Área habitable (sqft)", 500, 5000, 1500)
garage_cars = st.slider("Capacidad de garaje", 0, 4, 2)
garage_area = st.number_input("Área del garaje", 0, 1000, 300)
total_bsmt = st.number_input("Área del sótano", 0, 3000, 800)
year_built = st.number_input("Año construcción", 1900, 2025, 2000)

# crear dataframe
input_data = pd.DataFrame([default_values])

input_data["OverallQual"] = overall_qual
input_data["GrLivArea"] = gr_liv_area
input_data["GarageCars"] = garage_cars
input_data["GarageArea"] = garage_area
input_data["TotalBsmtSF"] = total_bsmt
input_data["YearBuilt"] = year_built

# predicción
if st.button("Predecir precio"):
    prediction = model.predict(input_data)
    st.success(f"Precio estimado: ${prediction[0]:,.2f}")