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


# ==========================================================
# 1. CARGA ROBUSTA DEL CSV (incluye ZIP específico)
# ==========================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Carga el dataset de housing probando varias rutas.
    Incluye la ruta específica:
    PruebaStreamlit/ML_EDA/housing_price_dataset.csv (1).zip

    Si no lo encuentra, permite al usuario subir el archivo
    (CSV o ZIP con un CSV dentro).
    """
    base_dir = Path(__file__).resolve().parent

    candidate_paths = [
        # Rutas clásicas de CSV
        base_dir / "housing_price_dataset.csv",
        base_dir / "housing_price_dataset_cleaned.csv",
        base_dir / "housing" / "housing_price_dataset.csv",
        base_dir.parent / "housing" / "housing_price_dataset.csv",
        base_dir.parent / "housing_price_dataset.csv",

        # Ruta específica que me has pasado (ajustada como relativa)
        base_dir / "ML_EDA" / "housing_price_dataset.csv (1).zip",
        base_dir.parent / "ML_EDA" / "housing_price_dataset.csv (1).zip",
        base_dir / "PruebaStreamlit" / "ML_EDA" / "housing_price_dataset.csv (1).zip",
    ]

    for path in candidate_paths:
        if path.exists():
            st.info(f"📄 Cargando dataset desde: `{path}`")
            if path.suffix == ".zip":
                # Lee el CSV directamente desde el ZIP
                return pd.read_csv(path, compression="zip")
            else:
                return pd.read_csv(path)

    # Si no se encuentra en disco, se pide subirlo manualmente
    st.warning(
        "⚠️ No se ha encontrado el archivo de datos en las rutas esperadas.\n\n"
        "Sube tu fichero `housing_price_dataset.csv` o un `.zip` que lo contenga para continuar."
    )

    uploaded = st.file_uploader(
        "Sube aquí tu `housing_price_dataset.csv` o `.zip`",
        type=["csv", "zip"]
    )

    if uploaded is not None:
        st.success("✅ Archivo subido correctamente, cargando datos...")
        if uploaded.name.endswith(".zip"):
            return pd.read_csv(uploaded, compression="zip")
        else:
            return pd.read_csv(uploaded)

    st.stop()  # Detiene la app hasta que haya datos


# ==========================================================
# 2. ENTRENAR MODELO XGBOOST + PIPELINE
# ==========================================================
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    - Aplica feature engineering.
    - Define preprocessor (ColumnTransformer).
    - Entrena un XGBoostRegressor dentro de un Pipeline.
    - Devuelve el pipeline entrenado y el split de test.
    """

    df_fe = df.copy()

    # ===== Feature engineering binario =====
    df_fe["Has_2plus_Bath"] = (df_fe["Bathrooms"] >= 2).astype(int)
    df_fe["Has_3plus_Bed"] = (df_fe["Bedrooms"] >= 3).astype(int)
    df_fe["New_House"] = (df_fe["YearBuilt"] >= 2000).astype(int)

    # ===== Definimos X e y =====
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

    # ===== Modelo XGBoost (mejor modelo) =====
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

    return pipe, X_train, X_test, y_train, y_test, X, y


# ==========================================================
# 3. APP STREAMLIT
# ==========================================================
def main():
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="🏠",
        layout="wide"
    )

    st.title("🏠 House Price Predictor (XGBoost + Streamlit)")

    st.markdown(
        """
        Esta app entrena un modelo de **XGBoost** sobre un dataset de viviendas
        y te permite introducir manualmente las características de una casa para
        estimar su **precio de venta**.

        Debajo puedes ver:
        - 📏 Métricas de error en test (*MAE, RMSE, R²*).
        - 🧩 Una “matriz de fallo” (precio real vs predicho, agrupados en rangos).
        - 📃 Una tabla con ejemplos de predicción y su error.
        - 📊 Visualizaciones básicas del dataset.
        """
    )

    # 1) Cargar datos
    df = load_data()

    # ==========================================================
    # 3.0. EXPLORACIÓN RÁPIDA DEL DATASET
    # ==========================================================
    st.header("👀 Exploración rápida del dataset")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Vista previa")
        st.dataframe(df.head(), use_container_width=True)

    with col_b:
        st.subheader("Descripción numérica")
        st.write(df.describe().T)

    st.markdown("### Distribuciones de variables clave")

    # Histograma Price
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Distribución de Price")
        fig1, ax1 = plt.subplots()
        ax1.hist(df["Price"], bins=30)
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Frecuencia")
        fig1.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.caption("Distribución de SquareFeet")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["SquareFeet"], bins=30)
        ax2.set_xlabel("SquareFeet")
        ax2.set_ylabel("Frecuencia")
        fig2.tight_layout()
        st.pyplot(fig2)

    with col3:
        st.caption("Distribución de YearBuilt")
        fig3, ax3 = plt.subplots()
        ax3.hist(df["YearBuilt"], bins=30)
        ax3.set_xlabel("YearBuilt")
        ax3.set_ylabel("Frecuencia")
        fig3.tight_layout()
        st.pyplot(fig3)

    st.markdown("### Relación entre tamaño y precio")

    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(df["SquareFeet"], df["Price"], alpha=0.5)
    ax_scatter.set_xlabel("SquareFeet")
    ax_scatter.set_ylabel("Price")
    ax_scatter.set_title("Price vs SquareFeet")
    fig_scatter.tight_layout()
    st.pyplot(fig_scatter)

    st.markdown("---")

    # 2) Entrenar modelo (se cachea)
    pipe, X_train, X_test, y_train, y_test, X_full, y_full = train_model(df)

    # RANGOS reales del dataset para SLIDERS
    min_sqft, max_sqft = int(df["SquareFeet"].min()), int(df["SquareFeet"].max())
    min_bed, max_bed = int(df["Bedrooms"].min()), int(df["Bedrooms"].max())
    min_bath, max_bath = int(df["Bathrooms"].min()), int(df["Bathrooms"].max())
    min_year, max_year = int(df["YearBuilt"].min()), int(df["YearBuilt"].max())
    neighborhoods = sorted(df["Neighborhood"].unique())

    # ==========================================================
    # 3.1. PREDICCIÓN MANUAL (SLIDERS + BINARIOS 0/1)
    # ==========================================================
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
    st.sidebar.subheader("Variables binarias (0 / 1)")

    # Checkboxes para 0/1 (permite control manual, pero por defecto coherentes)
    has_2plus_bath = st.sidebar.checkbox(
        "Has_2plus_Bath (baños ≥ 2)",
        value=(bathrooms >= 2)
    )
    has_3plus_bed = st.sidebar.checkbox(
        "Has_3plus_Bed (habitaciones ≥ 3)",
        value=(bedrooms >= 3)
    )
    new_house = st.sidebar.checkbox(
        "New_House (YearBuilt ≥ 2000)",
        value=(year_built >= 2000)
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Cuando estés listo, pulsa el botón para predecir.")

    # Construimos el DataFrame con exactamente las columnas esperadas
    input_dict = {
        "Neighborhood": [neighborhood],
        "SquareFeet": [sqft],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "YearBuilt": [year_built],
        "Has_2plus_Bath": [int(has_2plus_bath)],
        "Has_3plus_Bed": [int(has_3plus_bed)],
        "New_House": [int(new_house)],
    }
    input_df = pd.DataFrame(input_dict)

    col_pred, col_info = st.columns([2, 1])

    with col_pred:
        st.subheader("🔮 Predicción de precio para la vivienda introducida")

        if st.button("Calcular predicción"):
            pred_price = pipe.predict(input_df)[0]
            st.success(f"💰 Precio estimado: **{pred_price:,.0f}** (unidad monetaria del dataset)")
        else:
            st.info("Introduce los valores en la barra lateral y pulsa el botón para hacer una predicción.")

    with col_info:
        st.subheader("📊 Info rápida del dataset")
        st.metric("Nº de viviendas", len(df))
        st.metric("Precio medio", f"{df['Price'].mean():,.0f}")
        st.metric("M² medio", f"{df['SquareFeet'].mean():.1f}")

    st.markdown("---")

    # ==========================================================
    # 3.2. MÉTRICAS GLOBALES Y MATRIZ DE FALLO
    # ==========================================================
    st.header("📉 Rendimiento del modelo en el conjunto de test")

    y_pred_test = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)

    # Compatibilidad con versiones antiguas de sklearn: RMSE sin `squared=`
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (error medio abs.)", f"{mae:,.0f}")
    c2.metric("RMSE", f"{rmse:,.0f}")
    c3.metric("R² en test", f"{r2:.3f}")

    st.markdown("### 🔢 Matriz de fallo (precio real vs precio predicho, por rangos)")

    # Bineamos precios reales y predichos en cuartiles
    n_bins = 4
    true_bins = pd.qcut(y_test, q=n_bins, duplicates="drop")
    pred_bins = pd.qcut(y_pred_test, q=n_bins, duplicates="drop")

    conf_mat = pd.crosstab(true_bins, pred_bins)
    st.caption(
        "Filas: rangos de **precio real** · Columnas: rangos de **precio predicho**.\n"
        "Los valores son el número de viviendas en cada combinación."
    )
    st.dataframe(conf_mat, use_container_width=True)

    # Heatmap simple con matplotlib
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

    # ==========================================================
    # 3.3. TABLA DE ERRORES POR VIVIENDA
    # ==========================================================
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

