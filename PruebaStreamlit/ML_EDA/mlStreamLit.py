import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Keras / TensorFlow para la red neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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
# 2. ENTRENAR MODELO XGBOOST + PIPELINE (VENTANA 1)
# ==========================================================
@st.cache_resource
def train_model_xgb(df: pd.DataFrame):
    """
    - Aplica feature engineering.
    - Define preprocessor (ColumnTransformer).
    - Entrena un XGBRegressor dentro de un Pipeline.
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
            # En versiones nuevas de sklearn, el arg es sparse_output=False.
            # Si da error, cambia sparse_output por sparse=False.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
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
# 3. ENTRENAR MODELO RED NEURONAL (VENTANA 2)
# ==========================================================
@st.cache_resource
def train_model_nn(df: pd.DataFrame):
    """
    Implementa un modelo de Red Neuronal Profunda (similar a h2o.deeplearning)
    para predecir el precio de casas.

    - Split 80% / 20% -> train+valid / test
    - Del 80%, otro split 80% / 20% -> train / valid
      => 64% train, 16% valid, 20% test.
    - One-Hot para categóricas (Neighborhood).
    - Estandarización de las features (StandardScaler).
    - Arquitectura con varias capas ocultas + Dropout.
    - Optimización con MAE como función de pérdida principal.
    """

    df_fe = df.copy()

    # ===== Mismo feature engineering que para XGBoost =====
    df_fe["Has_2plus_Bath"] = (df_fe["Bathrooms"] >= 2).astype(int)
    df_fe["Has_3plus_Bed"] = (df_fe["Bedrooms"] >= 3).astype(int)
    df_fe["New_House"] = (df_fe["YearBuilt"] >= 2000).astype(int)

    X = df_fe.drop(columns=["Price"])
    y = df_fe["Price"].values

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
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # ---- 1º split: 80% train+valid, 20% test ----
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ---- 2º split: del 80%, 80% train y 20% valid ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,   # 20% de 80% => 16% del total
        random_state=42
    )

    # ---- One-Hot + numéricas ----
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # ---- Escalado global (num + dummies) ----
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_proc)
    X_val_sc = scaler.transform(X_val_proc)
    X_test_sc = scaler.transform(X_test_proc)

    input_dim = X_train_sc.shape[1]

    # ===== Definimos la arquitectura de la red =====
    model = Sequential()
    # Capa de entrada -> oculta 1
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.2))
    # Oculta 2
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    # Oculta 3
    model.add(Dense(32, activation="relu"))
    # Capa de salida (regresión)
    model.add(Dense(1, activation="linear"))

    # ===== Compilación =====
    # Usamos MAE como función de pérdida (como en tu memoria),
    # y mostramos MAE como métrica.
    model.compile(
        optimizer="adam",
        loss="mae",
        metrics=["mae"]
    )

    # ===== Entrenamiento =====
    # Mini-batch size = 32 (como recomiendan Masters & Luschi).
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_sc,
        y_train,
        validation_data=(X_val_sc, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Guardamos history.history (dict) porque el objeto History no es serializable.
    history_dict = history.history

    return {
        "model": model,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_sc": X_train_sc,
        "X_val_sc": X_val_sc,
        "X_test_sc": X_test_sc,
        "history": history_dict,
    }


# ==========================================================
# 4. PÁGINA XGBOOST
# ==========================================================
def page_xgboost(df: pd.DataFrame):
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

    # ==========================================================
    # 4.0. EXPLORACIÓN RÁPIDA DEL DATASET
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
    pipe, X_train, X_test, y_train, y_test, X_full, y_full = train_model_xgb(df)

    # RANGOS reales del dataset para SLIDERS
    min_sqft, max_sqft = int(df["SquareFeet"].min()), int(df["SquareFeet"].max())
    min_bed, max_bed = int(df["Bedrooms"].min()), int(df["Bedrooms"].max())
    min_bath, max_bath = int(df["Bathrooms"].min()), int(df["Bathrooms"].max())
    min_year, max_year = int(df["YearBuilt"].min()), int(df["YearBuilt"].max())
    neighborhoods = sorted(df["Neighborhood"].unique())

    # ==========================================================
    # 4.1. PREDICCIÓN MANUAL (SLIDERS + BINARIOS 0/1)
    # ==========================================================
    st.sidebar.header("🧮 Introduce los datos de la vivienda (XGBoost)")

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
        neighborhoods,
        key="xgb_neighborhood"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Variables binarias (0 / 1)")

    # Checkboxes para 0/1 (permite control manual, pero por defecto coherentes)
    has_2plus_bath = st.sidebar.checkbox(
        "Has_2plus_Bath (baños ≥ 2)",
        value=(bathrooms >= 2),
        key="xgb_has_2plus_bath"
    )
    has_3plus_bed = st.sidebar.checkbox(
        "Has_3plus_Bed (habitaciones ≥ 3)",
        value=(bedrooms >= 3),
        key="xgb_has_3plus_bed"
    )
    new_house = st.sidebar.checkbox(
        "New_House (YearBuilt ≥ 2000)",
        value=(year_built >= 2000),
        key="xgb_new_house"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Cuando estés listo, pulsa el botón para predecir (XGBoost).")

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
        st.subheader("🔮 Predicción de precio para la vivienda introducida (XGBoost)")

        if st.button("Calcular predicción (XGBoost)"):
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
    # 4.2. MÉTRICAS GLOBALES Y MATRIZ DE FALLO
    # ==========================================================
    st.header("📉 Rendimiento del modelo XGBoost en el conjunto de test")

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
    im = ax.imshow(conf_mat.values)

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
    ax.set_title("Matriz de fallo (true vs pred binned) - XGBoost")
    fig.tight_layout()
    st.pyplot(fig)

    # ==========================================================
    # 4.3. TABLA DE ERRORES POR VIVIENDA
    # ==========================================================
    st.markdown("### 📃 Ejemplos de predicción vs realidad (XGBoost)")

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


# ==========================================================
# 5. PÁGINA RED NEURONAL (DEEPL)
# ==========================================================
def page_nn(df: pd.DataFrame):
    st.title("🧠 House Price Predictor con Red Neuronal (Deep Learning)")

    st.markdown(
        """
        En esta ventana se entrena un modelo de **Red Neuronal Profunda** para
        predecir el precio de la vivienda, siguiendo la metodología descrita:

        - División de la base de datos en:
          - 64% datos de **entrenamiento**
          - 16% datos de **validación**
          - 20% datos de **test**
        - Codificación **One-Hot** de las variables categóricas (barrio).
        - Entrenamiento con descenso por gradiente y **mini-batch size = 32**.
        - Optimización del modelo usando el **MAE** (error absoluto medio).
        - Uso de **Early Stopping** para evitar overfitting.
        """
    )

    # Entrenamos / recuperamos el modelo cacheado
    res = train_model_nn(df)
    model = res["model"]
    preprocessor = res["preprocessor"]
    scaler = res["scaler"]
    X_test = res["X_test"]
    y_test = res["y_test"]
    X_test_sc = res["X_test_sc"]
    history = res["history"]

    # RANGOS reales del dataset para SLIDERS
    min_sqft, max_sqft = int(df["SquareFeet"].min()), int(df["SquareFeet"].max())
    min_bed, max_bed = int(df["Bedrooms"].min()), int(df["Bedrooms"].max())
    min_bath, max_bath = int(df["Bathrooms"].min()), int(df["Bathrooms"].max())
    min_year, max_year = int(df["YearBuilt"].min()), int(df["YearBuilt"].max())
    neighborhoods = sorted(df["Neighborhood"].unique())

    # ==========================================================
    # 5.1. PREDICCIÓN MANUAL PARA LA RED NEURONAL
    # ==========================================================
    st.sidebar.header("🧮 Introduce los datos de la vivienda (Red Neuronal)")

    sqft = st.sidebar.slider(
        "Superficie (SquareFeet) [NN]",
        min_value=min_sqft,
        max_value=max_sqft,
        value=int(np.median(df["SquareFeet"])),
        key="nn_sqft"
    )

    bedrooms = st.sidebar.slider(
        "Nº de habitaciones (Bedrooms) [NN]",
        min_value=min_bed,
        max_value=max_bed,
        value=int(np.median(df["Bedrooms"])),
        key="nn_bedrooms"
    )

    bathrooms = st.sidebar.slider(
        "Nº de baños (Bathrooms) [NN]",
        min_value=min_bath,
        max_value=max_bath,
        value=int(np.median(df["Bathrooms"])),
        key="nn_bathrooms"
    )

    year_built = st.sidebar.slider(
        "Año de construcción (YearBuilt) [NN]",
        min_value=min_year,
        max_value=max_year,
        value=int(np.median(df["YearBuilt"])),
        key="nn_year_built"
    )

    neighborhood = st.sidebar.selectbox(
        "Barrio (Neighborhood) [NN]",
        neighborhoods,
        key="nn_neighborhood"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Variables binarias (0 / 1) [NN]")

    has_2plus_bath = st.sidebar.checkbox(
        "Has_2plus_Bath (baños ≥ 2) [NN]",
        value=(bathrooms >= 2),
        key="nn_has_2plus_bath"
    )
    has_3plus_bed = st.sidebar.checkbox(
        "Has_3plus_Bed (habitaciones ≥ 3) [NN]",
        value=(bedrooms >= 3),
        key="nn_has_3plus_bed"
    )
    new_house = st.sidebar.checkbox(
        "New_House (YearBuilt ≥ 2000) [NN]",
        value=(year_built >= 2000),
        key="nn_new_house"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Cuando estés listo, pulsa el botón para predecir (Red Neuronal).")

    # Construimos el DataFrame de entrada
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
        st.subheader("🔮 Predicción de precio para la vivienda introducida (Red Neuronal)")

        if st.button("Calcular predicción (Red Neuronal)"):
            # Mismo preprocesado que en entrenamiento
            X_input_proc = preprocessor.transform(input_df)
            X_input_sc = scaler.transform(X_input_proc)
            pred_price = float(model.predict(X_input_sc).ravel()[0])
            st.success(f"💰 Precio estimado (NN): **{pred_price:,.0f}** (unidad monetaria del dataset)")
        else:
            st.info("Introduce los valores en la barra lateral y pulsa el botón para hacer una predicción (NN).")

    with col_info:
        st.subheader("📊 Info rápida del dataset")
        st.metric("Nº de viviendas", len(df))
        st.metric("Precio medio", f"{df['Price'].mean():,.0f}")
        st.metric("M² medio", f"{df['SquareFeet'].mean():.1f}")

    st.markdown("---")

    # ==========================================================
    # 5.2. CURVAS DE ENTRENAMIENTO (PÉRDIDA TRAIN vs VALID)
    # ==========================================================
    st.header("📉 Curvas de entrenamiento de la Red Neuronal")

    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    if loss and val_loss:
        fig, ax = plt.subplots()
        ax.plot(loss, label="Train loss (MAE)")
        ax.plot(val_loss, label="Valid loss (MAE)")
        ax.set_xlabel("Época")
        ax.set_ylabel("MAE")
        ax.set_title("Evolución de la pérdida (MAE) en entrenamiento y validación")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Si las curvas se separan mucho (valid peor que train), "
            "puede indicar **overfitting**, como se describe en tu memoria."
        )
    else:
        st.info("No se han podido recuperar las curvas de entrenamiento.")

    st.markdown("---")

    # ==========================================================
    # 5.3. MÉTRICAS EN TEST PARA LA RED NEURONAL
    # ==========================================================
    st.header("📉 Rendimiento del modelo de Red Neuronal en el conjunto de test")

    y_pred_test = model.predict(X_test_sc).ravel()

    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (error medio abs.) [NN]", f"{mae:,.0f}")
    c2.metric("RMSE [NN]", f"{rmse:,.0f}")
    c3.metric("R² en test [NN]", f"{r2:.3f}")

    st.markdown("### 📃 Ejemplos de predicción vs realidad (Red Neuronal)")

    comp_df = pd.DataFrame({
        "Precio_real": y_test,
        "Precio_predicho": y_pred_test,
    })
    comp_df["Error"] = comp_df["Precio_real"] - comp_df["Precio_predicho"]
    comp_df["Error_abs"] = comp_df["Error"].abs()

    st.dataframe(
        comp_df.sort_values("Error_abs", ascending=False).head(20),
        use_container_width=True
    )
    st.caption(
        "Se muestran las 20 viviendas donde la red neuronal más se equivoca (en valor absoluto). "
        "La métrica principal utilizada para optimizar el modelo es el **MAE**."
    )


# ==========================================================
# 6. APP STREAMLIT (SELECTOR DE VENTANA)
# ==========================================================
def main():
    st.set_page_config(
        page_title="House Price Predictor (XGBoost + NN)",
        page_icon="🏠",
        layout="wide"
    )

    # 1) Cargar datos
    df = load_data()

    # Selector de ventana en la barra lateral
    st.sidebar.title("🧭 Navegación")
    page = st.sidebar.radio(
        "Elige la ventana:",
        ("📈 Modelo XGBoost", "🧠 Red Neuronal (Deep Learning)")
    )

    if page == "📈 Modelo XGBoost":
        page_xgboost(df)
    else:
        page_nn(df)


if __name__ == "__main__":
    main()
