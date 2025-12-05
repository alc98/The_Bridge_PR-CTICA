import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import datetime

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
# Intenta cargar un logo; si no existe, usa un icono por defecto
try:
    logo = Image.open("logo.png")   # cambia por tu ruta de logo
    page_icon = logo
except Exception:
    logo = None
    page_icon = "🧠"

st.set_page_config(
    page_title="Brain MRI Tumor App",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============
# CARGA DE DATOS
# ==============
@st.cache_data
def load_data(path: str = "data.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame()  # vacío si no lo encuentra
    return df


df = load_data()

# =========================================================
# PÁGINA 1 – INTRO: warning, explicación cáncer + modelo
# =========================================================
def page_intro():
    st.title("🧠 Detección y segmentación de tumores cerebrales")

    if logo is not None:
        st.image(logo, width=120)

    st.markdown("## Introducción al problema clínico")
    st.warning(
        "El cáncer cerebral, y en particular los gliomas de bajo grado (LGG), "
        "requiere un diagnóstico precoz y una monitorización cuidadosa. "
        "Las resonancias magnéticas (MRI) permiten visualizar el tumor, "
        "pero la delimitación manual es lenta y dependiente del especialista."
    )

    st.markdown("## Nuestra propuesta de modelo")
    st.info(
        "En este proyecto utilizamos un modelo de **segmentación automática** "
        "entrenado sobre imágenes de resonancia. El modelo identifica, píxel a píxel, "
        "la región tumoral, generando una **máscara** que coloreamos en rojo sobre la MRI.\n\n"
        "**¿Por qué ayuda esto al problema?**\n"
        "- Reduce el tiempo de segmentación manual.\n"
        "- Aporta medidas cuantitativas (tamaño, porcentaje de corte ocupado).\n"
        "- Facilita el seguimiento de la evolución del tumor entre estudios."
    )

    st.markdown("---")
    st.markdown(
        "En las siguientes páginas podrás ver:\n"
        "- Estadísticas de la cohorte de pacientes.\n"
        "- Ejemplos de casos positivos y negativos.\n"
        "- Visualizaciones interactivas y contenido multimedia."
    )


# =========================================================
# PÁGINA 2 – DATAFRAME + GRÁFICAS (pie, barras, selectbox)
# =========================================================
def page_dataset():
    st.title("📊 Análisis de la base de datos")

    if df.empty:
        st.error("No se ha encontrado `data.csv`. Colócalo junto a `app.py` y recarga la página.")
        return

    st.subheader("Vista general del dataset")
    st.dataframe(df)

    # Asumimos que hay columnas 'Gender' y 'Tumor' (bool: 1=tumor, 0=no tumor)
    # Ajusta los nombres si tus columnas son otras.
    gender_col = "Gender"
    tumor_col = "Tumor"  # cambia por el nombre real: p.ej. "brain_tumor", "label", etc.

    if gender_col in df.columns:
        st.markdown("### Distribución por género")

        df_count = df.groupby(gender_col).size().reset_index(name="count")

        fig_pie = px.pie(
            df_count,
            values="count",
            names=gender_col,
            title="Distribución de pacientes por género"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Probabilidad media de tumor por género (si existe columna boolean / 0-1)
        if tumor_col in df.columns:
            st.markdown("### Probabilidad de tumor por género")

            # si tumor_col es 0/1 o bool, el mean() es la probabilidad
            df_avg = df.groupby(gender_col)[tumor_col].mean().reset_index(name="Tumor_Prob")

            fig_bar = px.bar(
                df_avg,
                x=gender_col,
                y="Tumor_Prob",
                title="Probabilidad media de tumor por género",
                labels={"Tumor_Prob": "Probabilidad de tumor"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Selectbox con géneros y mostrar la probabilidad
            st.markdown("### Consulta por género")
            genders = df[gender_col].dropna().unique().tolist()
            sel_gender = st.selectbox("Selecciona género", genders)

            prob_sel = df_avg.loc[df_avg[gender_col] == sel_gender, "Tumor_Prob"].values
            if len(prob_sel) > 0:
                st.success(
                    f"Probabilidad media estimada de tumor para **{sel_gender}**: "
                    f"**{prob_sel[0]*100:.2f}%**"
                )
        else:
            st.info(f"No se encontró la columna `{tumor_col}` para calcular probabilidad de tumor.")
    else:
        st.info(f"No se encontró la columna `{gender_col}` en el CSV.")

    # Gráfico de clases por bool (distribución global de tumor vs no tumor)
    if tumor_col in df.columns:
        st.markdown("### Distribución global de clases (tumor vs no tumor)")

        class_counts = df[tumor_col].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]

        fig_bool = px.bar(
            class_counts,
            x="Class",
            y="Count",
            title="Número de pacientes por clase (0 = no tumor, 1 = tumor)",
            text="Count"
        )
        st.plotly_chart(fig_bool, use_container_width=True)
    else:
        st.info(f"No se encontró la columna `{tumor_col}` para la gráfica de clases.")


# =========================================================
# PÁGINA 3 – CASOS POSITIVO/NEGATIVO CON MÁSCARA
# =========================================================
def page_cases():
    st.title("🖼️ Casos ejemplo: negativo vs positivo")

    st.markdown(
        "En esta sección mostramos un ejemplo de paciente **sin tumor** (caso negativo) "
        "y un paciente **con tumor** (caso positivo), junto con la máscara de segmentación."
    )

    # Cambia las rutas por tus imágenes reales:
    neg_img_path = "images/caso_negativo_mri.png"
    neg_mask_path = "images/caso_negativo_mask.png"
    pos_img_path = "images/caso_positivo_mri.png"
    pos_mask_path = "images/caso_positivo_mask.png"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Caso negativo (sin tumor)")
        try:
            neg_img = Image.open(neg_img_path)
            st.image(neg_img, caption="MRI – caso negativo", use_column_width=True)
        except Exception:
            st.info(f"Coloca la imagen del caso negativo en `{neg_img_path}`.")

        try:
            neg_mask = Image.open(neg_mask_path)
            st.image(neg_mask, caption="Máscara – caso negativo (sin tumor)", use_column_width=True)
        except Exception:
            st.info(f"Coloca la máscara del caso negativo en `{neg_mask_path}`.")

    with col2:
        st.subheader("Caso positivo (con tumor)")
        try:
            pos_img = Image.open(pos_img_path)
            st.image(pos_img, caption="MRI – caso positivo", use_column_width=True)
        except Exception:
            st.info(f"Coloca la imagen del caso positivo en `{pos_img_path}`.")

        try:
            pos_mask = Image.open(pos_mask_path)
            st.image(pos_mask, caption="Máscara – caso positivo (tumor en rojo)", use_column_width=True)
        except Exception:
            st.info(f"Coloca la máscara del caso positivo en `{pos_mask_path}`.")


# =========================================================
# PÁGINA 4 – MULTIMEDIA: FOTOS, VÍDEO, CITA
# =========================================================
def page_media():
    st.title("🎥 Demo visual y cita")

    st.subheader("Imágenes de ejemplo")
    # Imagen local
    try:
        img_local = Image.open("imagen.png")  # cambia a tu ruta
        st.image(img_local, caption="Imagen local de ejemplo", use_column_width=True)
    except Exception:
        st.info("Coloca una imagen llamada `imagen.png` junto a `app.py` o cambia la ruta.")

    # Imagen desde URL (solo demostración)
    st.image("https://picsum.photos/1280", caption="Imagen de ejemplo desde URL", use_column_width=True)

    st.subheader("Vídeo demostrativo de la app / modelo")
    # Vídeo local
    try:
        with open("video.mp4", "rb") as video_file:   # cambia a tu ruta
            video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception:
        st.info("Coloca un `video.mp4` junto a `app.py` o actualiza la ruta en el código.")

    st.subheader("📅 Simulación de cita")
    cita = st.date_input("Selecciona una fecha para la cita de revisión", datetime.date.today())
    st.success(f"Fecha seleccionada: {cita.strftime('%d/%m/%Y')}")


# =========================
# MENÚ PRINCIPAL (SIDEBAR)
# =========================
def main():
    st.title("Brain MRI Tumor – Demo Streamlit")

    menu = [
        "🏠 Introducción",
        "📊 Base de datos y gráficas",
        "🖼️ Casos ejemplo",
        "🎥 Multimedia y cita"
    ]

    choice = st.sidebar.selectbox("Navegación", menu)

    if choice == "🏠 Introducción":
        page_intro()
    elif choice == "📊 Base de datos y gráficas":
        page_dataset()
    elif choice == "🖼️ Casos ejemplo":
        page_cases()
    elif choice == "🎥 Multimedia y cita":
        page_media()


if __name__ == "__main__":
    main()
