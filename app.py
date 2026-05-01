import streamlit as st
import pandas as pd
import joblib
import numpy as np

# cargar modelo
model = joblib.load("modelo_pipeline.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 Predicción de precios de casas")

unit = st.radio("Selecciona unidad de medida:", ["sqft", "m²"])

st.markdown("Ingresa las características de la casa:")

# UI mejorada
# factor de conversión
SQFT_TO_M2 = 0.092903
M2_TO_SQFT = 10.7639

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Calidad general", 1, 10, 5)

    gr_liv_area_input = st.number_input(
        f"Área habitable ({unit})",
        30 if unit == "m²" else 300,
        500 if unit == "m²" else 5000,
        100 if unit == "m²" else 1000
    )

    total_bsmt_input = st.number_input(
        f"Área sótano ({unit})",
        0,
        300 if unit == "m²" else 3000,
        50 if unit == "m²" else 500
    )

with col2:
    garage_cars = st.slider("Capacidad garaje", 0, 4, 2)

    garage_area_input = st.number_input(
        f"Área garaje ({unit})",
        0,
        100 if unit == "m²" else 1000,
        20 if unit == "m²" else 200
    )

    year_built = st.slider("Año construcción", 1900, 2025, 2000)

    # convertir a sqft si el usuario usa m²
if unit == "m²":
    gr_liv_area = gr_liv_area_input * M2_TO_SQFT
    total_bsmt = total_bsmt_input * M2_TO_SQFT
    garage_area = garage_area_input * M2_TO_SQFT
else:
    gr_liv_area = gr_liv_area_input
    total_bsmt = total_bsmt_input
    garage_area = garage_area_input


# dataframe input
input_data = pd.DataFrame([{col: None for col in columns}])

# sobrescribir con valores reales
input_data["OverallQual"] = overall_qual
input_data["GrLivArea"] = gr_liv_area
input_data["TotalBsmtSF"] = total_bsmt
input_data["GarageCars"] = garage_cars
input_data["GarageArea"] = garage_area
input_data["YearBuilt"] = year_built


# predicción
if st.button("Predecir precio"):
    prediction = model.predict(input_data)[0]

    # rango de precios
    mae = 17726  # usa tu MAE real
    lower = prediction - mae
    upper = prediction + mae

    st.success(f"💰 Precio estimado: ${prediction:,.0f}")

    st.info(f"📊 Rango probable: ${lower:,.0f} - ${upper:,.0f}")

if st.checkbox("Ver importancia de variables"):
        importances = model.named_steps["model"].feature_importances_
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).head(10)

        st.bar_chart(df_imp.set_index("feature"))