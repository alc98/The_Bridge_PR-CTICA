import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import matplotlib.pyplot as plt


# ===========================
# 1. Carga de datos
# ===========================
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Carga el CSV limpio de casas.
    Ajusta la ruta si lo tienes en otra carpeta.
    """
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "housing_price_dataset.csv"  # adapta si lo tienes en /housing/...
    df = pd.read_csv(path)
    return df


# ===========================
# 2. Entrenamiento del modelo (XGBoost) + split
# ===========================
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Feature engineering, split train/test y entrenamiento de XGBRegressor
    dentro de un Pipeline con ColumnTransformer.
    Devuelve:
      - pipeline entrenado
      - X_test, y_test para evaluar errores
    """

    # --- Feature engineering ---
    df_fe = df.copy()
    df_fe["Has_2plus_Bath"] = (df_fe["Bathrooms"] >= 2).astype(int)
    df_fe["Has_3plus_Bed"] = (df_fe["Bedrooms"] >= 3).astype(int)
    df_fe["New_House"] = (df_fe["YearBuilt"] >= 2000).astype(int)

    X = df_fe.drop(columns=["Price"])
    y = df_fe["Price"]

    numeric_features = [
        "SquareFeet",
        "Bedrooms",
        "Bathrooms",
        "YearBuilt",
        "Has_2plus_Bath",
        "Has_3plus_Bed",
        "New_House",
    ]
    categorical_features = ["Neighborhood"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # --- Modelo "mejor" (XGBRegressor, parámetros razonables) ---
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        reg_lambda=1.0,
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    return pipe, X_test, y_test, X, y


# ===========================
# 3. App Streamlit
# ===========================
def main():
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="🏠",
        layout="wide"
    )

    st.title("🏠 House Price Predictor (XGBoost + Streamlit)")

    st.markdown(
        """
        Esta app entrena un modelo de **XGBoost** sobre el dataset de viviendas
        y te permite introducir manualmente las características de una casa para
        estimar su **precio de venta**.  
        
        Debajo, se muestran:
        - Métricas de error en test (MAE, RMSE, R²).  
        - Una “matriz de fallo” (precio real vs predicho, binned).  
        - Una tabla con ejemplos de predicción y su error.
        """
    )

    # 1) Cargar datos y modelo
    df = load_data()
    pipe, X_test, y_test, X_full, y_full = train_model(df)

    # Para los rangos de los sliders usamos el df original
    min_sqft, max_sqft = int(df["SquareFeet"].min()), int(df["SquareFeet"].max())
    min_bed, max_bed = int(df["Bedrooms"].min()), int(df["Bedrooms"].max())
    min_bath, max_bath = int(df["Bathrooms"].min()), int(df["Bathrooms"].max())
    min_year, max_year = int(df["YearBuilt"].min()), int(df["YearBuilt"].max())
    neighborhoods = sorted(df["Neighborhood"].unique())

    st.sidebar.header("🧮 Introduce los datos de la vivienda")

    sqft = st.sidebar.slider(
        "Superficie (SquareFeet)",
        min_value=min_sqft,
        max_value=max_sqft,
        value=int(np.median(df["SquareFeet"]))
    )

    bedrooms = st.sidebar.slider(
        "Nº de habitaciones (Bedrooms)",
        min_value=min_bed,
        max_value=max_bed,
        value=int(np.median(df["Bedrooms"]))
    )

    bathrooms = st.sidebar.slider(
        "Nº de baños (Bathrooms)",
        min_value=min_bath,
        max_value=max_bath,
        value=int(np.median(df["Bathrooms"]))
    )

    year_built = st.sidebar.slider(
        "Año de construcción (YearBuilt)",
        min_value=min_year,
        max_value=max_year,
        value=int(np.median(df["YearBuilt"]))
    )

    neighborhood = st.sidebar.selectbox(
        "Barrio (Neighborhood)",
        neighborhoods
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Cuando estés listo, pulsa el botón para predecir.")

    # Construimos el dataframe de 1 fila con los mismos nombres de columnas que X
    input_dict = {
        "SquareFeet": [sqft],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "YearBuilt": [year_built],
        "Neighborhood": [neighborhood],
        # Estas columnas NO las necesita el pipeline como entrada,
        # porque se recalculan en train_model(). Aquí puedes omitirlas,
        # o poner los mismos cálculos si quieres consistencia explícita.
        "Has_2plus_Bath": [(bathrooms >= 2)],
        "Has_3plus_Bed": [(bedrooms >= 3)],
        "New_House": [(year_built >= 2000)],
    }

    # Pero ojo: el preprocessor creado en train_model solo espera:
    # ["Neighborhood"] + numeric_features.
    # Así que nos quedamos solo con esas columnas:
    input_df = pd.DataFrame(input_dict)[[
        "Neighborhood",
        "SquareFeet",
        "Bedrooms",
        "Bathrooms",
        "YearBuilt",
        "Has_2plus_Bath",
        "Has_3plus_Bed",
        "New_House",
    ]]

    col_pred, col_info = st.columns([2, 1])

    with col_pred:
        st.subheader("🔮 Predicción de precio para la vivienda introducida")

        if st.button("Calcular predicción"):
            pred_price = pipe.predict(input_df)[0]
            st.success(f"Precio estimado: **{pred_price:,.0f}** unidades monetarias")
            st.caption("(*La moneda depende de cómo esté definido tu dataset: €, $, etc.*)")
        else:
            st.info("Introduce los valores en la barra lateral y pulsa el botón para predecir.")

    with col_info:
        st.subheader("📊 Info rápida del dataset")
        st.metric("Nº de viviendas", len(df))
        st.metric("Precio medio", f"{df['Price'].mean():,.0f}")
        st.metric("M² medio", f"{df['SquareFeet'].mean():.1f}")

    st.markdown("---")
    st.header("📉 Rendimiento del modelo en el conjunto de test")

    # ===========================
    #  Métricas + matriz de fallo
    # ===========================
    y_pred_test = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    r2 = r2_score(y_test, y_pred_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (error medio abs.)", f"{mae:,.0f}")
    c2.metric("RMSE", f"{rmse:,.0f}")
    c3.metric("R² en test", f"{r2:.3f}")

    st.markdown("### 🔢 Matriz de fallo (precio real vs precio predicho, binned)")

    # Bineamos precios reales y predichos en cuartiles
    # (puedes ajustar el nº de bins si quieres más detalle)
    n_bins = 4
    true_bins = pd.qcut(y_test, q=n_bins, duplicates="drop")
    pred_bins = pd.qcut(y_pred_test, q=n_bins, duplicates="drop")

    conf_mat = pd.crosstab(true_bins, pred_bins)
    st.caption(
        "Filas: rango de precios **reales**, "
        "Columnas: rango de precios **predichos**. "
        "Los valores son conteos de casas en cada combinación."
    )
    st.dataframe(conf_mat, use_container_width=True)

    # Heatmap simple
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(conf_mat.values, cmap="Blues")

    ax.set_xticks(range(len(conf_mat.columns)))
    ax.set_xticklabels([str(c) for c in conf_mat.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(conf_mat.index)))
    ax.set_yticklabels([str(i) for i in conf_mat.index])

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, conf_mat.values[i, j],
                    ha="center", va="center", color="black", fontsize=8)

    ax.set_xlabel("Precio predicho (bins)")
    ax.set_ylabel("Precio real (bins)")
    ax.set_title("Matriz de fallo (true vs pred binned)")
    fig.tight_layout()
    st.pyplot(fig)

    # ===========================
    #  Tabla de ejemplos: real, predicho, error
    # ===========================
    st.markdown("### 📃 Ejemplos de predicción vs realidad")

    comp_df = pd.DataFrame({
        "Precio_real": y_test.values,
        "Precio_predicho": y_pred_test,
    })
    comp_df["Error"] = comp_df["Precio_real"] - comp_df["Precio_predicho"]
    comp_df["Error_abs"] = comp_df["Error"].abs()

    st.dataframe(
        comp_df.sort_values("Error_abs", ascending=False).head(20),
        use_container_width=True
    )
    st.caption("Se muestran las 20 viviendas donde el modelo más se equivoca (en valor absoluto).")


if __name__ == "__main__":
    main()
