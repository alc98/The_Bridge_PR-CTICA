import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
import matplotlib.pyplot as plt


# ===========================
# 1. Carga de datos
# ===========================
@st.cache_data
def load_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "housing_price_dataset.csv"
    return pd.read_csv(path)


# ===========================
# 2. Feature engineering + preprocessor
# ===========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()
    df_fe["Has_2plus_Bath"] = (df_fe["Bathrooms"] >= 2).astype(int)
    df_fe["Has_3plus_Bed"] = (df_fe["Bedrooms"] >= 3).astype(int)
    df_fe["New_House"] = (df_fe["YearBuilt"] >= 2000).astype(int)
    return df_fe


def build_preprocessor():
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

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )


# ===========================
# 3. Entrenamiento de modelos (LinearRegression y XGBoost) + split
# ===========================
@st.cache_resource
def train_models(df: pd.DataFrame, seed: int = 42):
    df_fe = add_features(df)

    X = df_fe.drop(columns=["Price"])
    y = df_fe["Price"]

    preprocessor = build_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Modelo 1: Linear Regression
    lr_model = LinearRegression()
    lr_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", lr_model),
    ])
    lr_pipe.fit(X_train, y_train)

    # Modelo 2: XGBoost
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        reg_lambda=1.0,
    )
    xgb_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model),
    ])
    xgb_pipe.fit(X_train, y_train)

    return {
        "LinearRegression": lr_pipe,
        "XGBRegressor": xgb_pipe,
    }, X_test, y_test


# ===========================
# 4. App Streamlit
# ===========================
def main():
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="",
        layout="wide"
    )

    st.title("House Price Predictor")

    st.markdown(
        """
        Esta app entrena dos modelos sobre el dataset de viviendas:
        - LinearRegression
        - XGBoost (XGBRegressor)

        Permite introducir características de una vivienda para estimar su precio y
        muestra métricas de rendimiento en test (MAE, RMSE, R²) y una matriz de fallo por bins.
        """
    )

    df = load_data()
    models, X_test, y_test = train_models(df)

    model_name = st.sidebar.selectbox("Modelo", list(models.keys()))
    pipe = models[model_name]

    min_sqft, max_sqft = int(df["SquareFeet"].min()), int(df["SquareFeet"].max())
    min_bed, max_bed = int(df["Bedrooms"].min()), int(df["Bedrooms"].max())
    min_bath, max_bath = int(df["Bathrooms"].min()), int(df["Bathrooms"].max())
    min_year, max_year = int(df["YearBuilt"].min()), int(df["YearBuilt"].max())
    neighborhoods = sorted(df["Neighborhood"].unique())

    st.sidebar.header("Introduce los datos de la vivienda")

    sqft = st.sidebar.slider("Superficie (SquareFeet)", min_sqft, max_sqft, int(np.median(df["SquareFeet"])))
    bedrooms = st.sidebar.slider("Nº de habitaciones (Bedrooms)", min_bed, max_bed, int(np.median(df["Bedrooms"])))
    bathrooms = st.sidebar.slider("Nº de baños (Bathrooms)", min_bath, max_bath, int(np.median(df["Bathrooms"])))
    year_built = st.sidebar.slider("Año de construcción (YearBuilt)", min_year, max_year, int(np.median(df["YearBuilt"])))
    neighborhood = st.sidebar.selectbox("Barrio (Neighborhood)", neighborhoods)

    input_dict = {
        "SquareFeet": [sqft],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "YearBuilt": [year_built],
        "Neighborhood": [neighborhood],
        "Has_2plus_Bath": [(bathrooms >= 2)],
        "Has_3plus_Bed": [(bedrooms >= 3)],
        "New_House": [(year_built >= 2000)],
    }
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
        st.subheader("Predicción de precio")
        if st.button("Calcular predicción"):
            pred_price = pipe.predict(input_df)[0]
            st.success(f"Precio estimado: {pred_price:,.0f} unidades monetarias")
        else:
            st.info("Introduce valores y pulsa el botón para predecir.")

    with col_info:
        st.subheader("Info rápida del dataset")
        st.metric("Nº de viviendas", len(df))
        st.metric("Precio medio", f"{df['Price'].mean():,.0f}")
        st.metric("M² medio", f"{df['SquareFeet'].mean():.1f}")

    st.markdown("---")
    st.header("Rendimiento del modelo en test")

    y_pred_test = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    r2 = r2_score(y_test, y_pred_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.0f}")
    c2.metric("RMSE", f"{rmse:,.0f}")
    c3.metric("R²", f"{r2:.3f}")

    st.markdown("### Matriz de fallo (precio real vs precio predicho, por bins)")

    n_bins = 4
    true_bins = pd.qcut(y_test, q=n_bins, duplicates="drop")
    pred_bins = pd.qcut(y_pred_test, q=n_bins, duplicates="drop")
    conf_mat = pd.crosstab(true_bins, pred_bins)

    st.dataframe(conf_mat, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(conf_mat.values, cmap="Blues")

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

    st.markdown("### Ejemplos de predicción vs realidad")

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


if __name__ == "__main__":
    main()
