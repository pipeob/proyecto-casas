import streamlit as st
import pandas as pd
import joblib
import numpy as np

# cargar modelo
model = joblib.load("modelo_pipeline.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 Predicción de precios de casas")

st.markdown("Ingresa las características de la casa:")

# UI mejorada
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Calidad general", 1, 10, 5)
    gr_liv_area = st.number_input("Área habitable (sqft)", 300, 5000, 1000)
    total_bsmt = st.number_input("Área sótano (sqft)", 0, 3000, 500)

with col2:
    garage_cars = st.slider("Capacidad garaje", 0, 4, 2)
    garage_area = st.number_input("Área garaje (sqft)", 0, 1000, 200)
    year_built = st.slider("Año construcción", 1900, 2025, 2000)

# conversión sqft → m2
st.markdown(f"📐 Área habitable en m²: **{gr_liv_area * 0.092903:.2f} m²**")


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