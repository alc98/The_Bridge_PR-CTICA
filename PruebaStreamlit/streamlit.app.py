import streamlit as st  
import pandas as pd
import plotly.express as px
from PIL import Image
import datetime
import requests
import base64
import io
import numpy as np
import random
from pathlib import Path
from PIL import Image



# Carpeta donde está el propio streamlit.app.py
BASE_DIR = Path(__file__).resolve().parent

# Carpeta de imágenes (Imagen está dentro de PruebaStreamlit junto al script)
IMAGES_DIR = BASE_DIR / "Imagen"   # ⬅️ AQUÍ, no hace falta "PruebaStreamlit" porque ya estás dentro

try:
    logo = Image.open("logo.png")
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

@st.cache_data
def load_data(path: str = "data.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame()
    return df

df = load_data()

GENDER_COL = "Gender"
TUMOR_COL = "Tumor"

def call_flask_model(api_url: str, pil_image: Image.Image):
    pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    url = api_url.rstrip("/") + "/predict"

    resp = requests.post(
        url,
        json={"image_base64": img_b64},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()

# 👇 AQUÍ PEGAS ESTO
try:
    from flask_app import app as backup_flask_app  # ajusta flask_app al nombre real
except Exception:
    backup_flask_app = None


def call_flask_model_backup(pil_image: Image.Image):
    if backup_flask_app is None:
        raise RuntimeError("Backup Flask app (.app) is not available or could not be imported.")

    pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    with backup_flask_app.test_client() as client:
        resp = client.post(
            "/predict",
            json={"image_base64": img_b64}
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Backup Flask app (.app) returned status {resp.status_code}: {resp.data!r}"
            )

        data = resp.get_json()
        if data is None:
            raise RuntimeError("Backup Flask app (.app) did not return valid JSON.")
        return data

def decode_mask_from_b64(mask_b64: str) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes))
    return np.array(mask_img)

def page_intro():
    st.header("🧠 Brain tumor detection and segmentation")
    
    try:
        mri_img = Image.open("images/TCGA_CS_4942_19970222_10.tif")
        st.image(
            mri_img,
            caption="Example brain MRI (TCGA_CS_4942_19970222_10)",
            use_column_width=True
        )
    except Exception:
        st.info(
            "Place the MRI image at `images/TCGA_CS_4942_19970222_10.tif` "
            "or update the path in `page_intro()`."
        )
    
    st.error(
        "- Around 80% of people living with a brain tumor require neurorehabilitation.\n"
        "- In Spain, more than 5,000 new brain tumor cases are diagnosed every year.\n"
        "- Brain tumors account for approximately 2% of all cancers diagnosed in adults and 15% of those diagnosed in children.\n"
        "- About 80% of patients will present cognitive dysfunction, and 78% will present motor dysfunction.\n"
        "- Therapeutic exercise can reduce cancer-related mortality by up to 59%."
    )

    if logo is not None:
        st.image(logo, width=120)

    st.markdown("## Introduction to the clinical problem")
    st.warning(
        "Brain cancer, and in particular low-grade gliomas (LGG), "
        "requires early diagnosis and careful monitoring. "
        "Magnetic resonance imaging (MRI) allows us to visualize the tumor, "
        "but manual delineation is slow and highly dependent on the specialist."
    )

    st.markdown("---")
    st.markdown(
        "In this demo you will see:\n"
        "- Statistics of the patient cohort.\n"
        "- Example positive and negative cases.\n"
        "- How a deep learning model can predict whether an MRI contains a tumor."
    )

    st.info(
        """
        From a clinical perspective, low-grade gliomas often affect relatively young
        adults and may present with seizures, headaches or subtle cognitive changes.
        Even though they are classified as "low grade", they can progress to
        high-grade gliomas, so longitudinal monitoring with MRI and, when indicated,
        histopathological and molecular analysis (e.g. IDH mutation, 1p/19q codeletion)
        are key for prognosis and treatment planning.
        """
    )

    st.markdown(
        """
        For radiologists and data scientists, MRI is interesting because it combines:
        - **Anatomical detail** (T1- and T2-weighted sequences).
        - **Edema and tumor extent** visualization (FLAIR).
        - In some protocols, **functional information** such as diffusion and perfusion,
          which can correlate with cell density and vascularity.
        Integrating these heterogeneous sources of information is one of the main
        motivations for using deep learning in neuro-oncology.
        """
    )
    
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent

def page_dataset():  
    st.header("📊 Dataset analysis")

    # 1st path: Imagen folder (current app)
    route_path_1 = BASE_DIR / "Imagen" / "route_label.csv"
    # 2nd path: brain-tumor-detection/data repo (e.g. develop branch)
    route_path_2 = BASE_DIR / "brain-tumor-detection" / "data" / "route_label.csv"

    df_routes = None
    used_path = None

    # First attempt
    try:
        df_routes = pd.read_csv(route_path_1)
        used_path = route_path_1
    except FileNotFoundError:
        # Second attempt
        try:
            df_routes = pd.read_csv(route_path_2)
            used_path = route_path_2
        except FileNotFoundError:
            st.error(
                "The file `route_label.csv` was not found in any of the paths:\n\n"
                f"- `{route_path_1}`\n"
                f"- `{route_path_2}`\n\n"
                "Please check your folder structure or the repository branch."
            )
            return

    # Drop index column if it exists
    if "Unnamed: 0" in df_routes.columns:
        df_routes = df_routes.drop(columns=["Unnamed: 0"])

    st.caption(
        f"Rows: {df_routes.shape[0]} · Columns: {df_routes.shape[1]}  "
        f"· CSV loaded from: `{used_path}`"
    )

    tab_table, tab_plots = st.tabs(["📄 Table", "📈 Plots"])




    # ===== PLOTS =====
    with tab_plots:
        if "mask" not in df_routes.columns:
            st.info("The CSV does not contain a `mask` column.")
            return

        st.subheader("Class distribution (0 = negative, 1 = positive)")

        # 1) Count 0 and 1
        class_counts = df_routes["mask"].value_counts().reset_index()
        class_counts.columns = ["mask_value", "Number of images"]

        # 2) Map to readable labels
        class_counts["Class"] = class_counts["mask_value"].map({
            0: "0 – Negative (no tumor)",
            1: "1 – Positive (tumor present)",
        })

        # 3) Keep only the columns needed for the plot
        class_counts = class_counts[["Class", "Number of images"]]

        # 4) Pie chart
        fig_pie = px.pie(
            class_counts,
            names="Class",
            values="Number of images",
            title="Class distribution: tumor vs no tumor",
        )

        # Center the chart using 3 columns
        col1, col2, col3 = st.columns(3)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)

        # Global prevalence (image-level)
        prevalence = df_routes["mask"].mean()
        st.markdown(
            f"**Global tumor prevalence (image level):** ≈ **{prevalence*100:.2f}%** "
            "of the images are labelled as positive (`mask = 1`)."
        )

        # ===== TABLE =====
        with tab_table:
            st.subheader("Overview of `route_label.csv`")
            st.dataframe(df_routes[df_routes.columns[]])
    # =====================================================================
    #  🔬 Scientific medical + data science interpretation
    # =====================================================================
    prevalence_global = df_routes["mask"].mean()
    negative_pct = (1 - prevalence_global) * 100
    positive_pct = prevalence_global * 100

    st.markdown(f"""

## 🧠 Scientific interpretation of the dataset

### 1. Cohort composition (image-level class distribution)

In this dataset:

- **≈ {negative_pct:.1f}%** of MRI slices are labelled as  
  **0 – Negative (no tumor)**  
- **≈ {positive_pct:.1f}%** of MRI slices are labelled as  
  **1 – Positive (tumor present)**  

This yields an **image-level tumor prevalence of approximately {positive_pct:.1f}%**.

From a methodological standpoint, this indicates a **moderately imbalanced dataset**, 
with a dominant negative class and a substantial proportion of positive slices.  
Therefore, **any classification model** must outperform a trivial baseline predicting 
the majority class (≈ **{negative_pct:.1f}% accuracy**) to demonstrate meaningful discriminative value.



### 2. Clinical and machine-learning implications

- The enrichment in tumor-positive slices (≈ {positive_pct:.1f}%) is higher than in routine clinical cohorts, 
  which usually contain far fewer tumors. This is advantageous for model development, as it provides a 
  sufficient number of positive examples to learn tumor-related patterns and to train segmentation models.

- Because of the moderate class imbalance, evaluation should not rely solely on accuracy. More informative metrics are:  
  - **Sensitivity / recall** for positive cases (`mask = 1`)  
  - **Specificity** for negative cases (`mask = 0`)  
  - **AUC-ROC** and **AUC-PR**, which better capture performance under imbalance.

- If the model tends to under-detect tumors, one may consider:  
  - **Class-weighted loss functions**  
  - **Focal loss**  
  - **Oversampling of positive slices** or undersampling of negatives.


### 3. Utility of the `mask` column

Although voxel-wise segmentation masks are available via `mask_path`, the binary image-level label (`mask`) enables:

- Rapid assessment of **class distribution** (as visualised in the pie chart).  
- Training of a **binary tumor vs. no-tumor classifier** as a screening or pre-filtering stage.  
- Stratified analyses, for example comparing intensity distributions or radiomic features between positive and negative slices.

From a clinical research perspective, the cohort can be succinctly described as:

> *"In this dataset, approximately {positive_pct:.1f}% of MRI slices contain visible tumor tissue according to expert segmentation. This prevalence establishes the baseline that any automated detection model must exceed in order to be clinically relevant."*

""")





def page_cases(): 
    st.header("🖼️ Ejemplos de tumores cerebrales en RM")

    st.markdown(
        """
        Aquí mostramos cortes de **resonancia magnética cerebral** con y sin **tumor segmentado**.
        En cada ejemplo verás:

        1. **RM original**  
        2. **Máscara binaria del tumor** (blanco = tumor, negro = fondo)  
        3. **RM con la máscara superpuesta** (solo en los casos con tumor)
        """
    )

    rows_dir = IMAGES_DIR

    # ------------------ CASOS CON TUMOR (row_*.png) ------------------
    tumor_rows = sorted(rows_dir.glob("row_*.png"))

    # ------------------ CASOS SIN TUMOR (example_no_tumor*.png) ------------------
    no_tumor_rows = sorted(rows_dir.glob("example_no_tumor*.png"))

    if not tumor_rows and not no_tumor_rows:
        st.error(
            "No se han encontrado imágenes ni `row_*.png` (con tumor) "
            "ni `example_no_tumor*.png` (sin tumor) en la carpeta "
            f"`{rows_dir}`."
        )
        return

    # =========================
    # Contenedor central
    # =========================
    left_empty, center, right_empty = st.columns([1, 4, 1])
    with center:
        st.markdown("<br>", unsafe_allow_html=True)

        # Primero SIN tumor, luego CON tumor
        tipo = st.radio(
            "Selecciona tipo de caso",
            ("🟢 Sin tumor", "🔴 Con tumor"),
            horizontal=True,
            index=0,
        )

        if tipo == "🔴 Con tumor":
            active_rows = tumor_rows
            state_key = "random_row_idx_tumor"
            boton_texto = "🔀 Mostrar otro caso con tumor"
            titulo_prefix = "Caso"
            subtitulo_suffix = "tumor cerebral segmentado"
        else:
            active_rows = no_tumor_rows
            state_key = "random_row_idx_no_tumor"
            boton_texto = "🔀 Mostrar otro caso sano"
            titulo_prefix = "Caso sano"
            subtitulo_suffix = "RM sin tumor visible"

        if not active_rows:
            if tipo == "🔴 Con tumor":
                st.warning(
                    "No hay imágenes `row_*.png` para casos con tumor.\n"
                    "Asegúrate de que `row_01.png`, `row_02.png`, ... están en la carpeta Imagen."
                )
            else:
                st.warning(
                    "No hay imágenes `example_no_tumor*.png` para casos sin tumor.\n"
                    "Coloca archivos como `example_no_tumor.png`, "
                    "`example_no_tumor2.png`, ... en la carpeta Imagen."
                )
            return

        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        # Botón centrado
        bc1, bc2, bc3 = st.columns([1, 2, 1])
        with bc2:
            if st.button(boton_texto):
                st.session_state[state_key] = random.randrange(len(active_rows))

        st.markdown("<br>", unsafe_allow_html=True)

        current_idx = st.session_state[state_key]
        current_path = active_rows[current_idx]

        stem = current_path.stem
        num_part = "".join(ch for ch in stem if ch.isdigit())
        case_number = num_part if num_part else "–"

        st.markdown(
            f"<h3 style='text-align:center'>{titulo_prefix} {case_number}: {subtitulo_suffix}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # =========================
        # Mostrar imágenes
        # =========================
        if tipo == "🔴 Con tumor":
            # fila row_XX con 3 columnas en una misma imagen
            img_row = Image.open(current_path)
            w, h = img_row.size
            col_w = w // 3

            img_mri      = img_row.crop((0,        0, col_w,   h))
            img_mask     = img_row.crop((col_w,    0, 2*col_w, h))
            img_mri_mask = img_row.crop((2*col_w,  0, w,       h))

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    "<h5 style='text-align:center'>RM original</h5>",
                    unsafe_allow_html=True,
                )
                st.image(img_mri, use_column_width=True)

            with c2:
                st.markdown(
                    "<h5 style='text-align:center'>Máscara de tumor</h5>",
                    unsafe_allow_html=True,
                )
                st.image(img_mask, use_column_width=True)

            with c3:
                st.markdown(
                    "<h5 style='text-align:center'>RM con máscara</h5>",
                    unsafe_allow_html=True,
                )
                st.image(img_mri_mask, use_column_width=True)

            # 🧠 Descripción clínico–analítica en inglés (con tumor)
            st.markdown(
                """
                #### Clinical / data analyst interpretation (with tumor)

                - **Region of interest:** a focal hyperintense lesion is visible within the brain
                  parenchyma. The binary mask highlights all pixels classified as tumor.
                - **Segmentation concept:** every white pixel in the mask corresponds to voxels
                  that the model (or the manual annotation) considers part of the tumor.
                - **Visual benefit:** the overlaid image makes it easier to appreciate tumor
                  borders, mass effect and relationship to surrounding tissue.
                - **From a data point of view:** this slice would be labelled as a **positive
                  sample**, and the mask provides dense supervision for training segmentation
                  models (Dice, IoU, pixel-wise accuracy, etc.).
                """
            )

        else:
            # ejemplo sin tumor: una única RM; la máscara está ya implícita (vacía)
            img_mri = Image.open(current_path).convert("RGB")

            # Pequeño toggle de vista, pero la imagen es la misma
            vista = st.radio(
                "Vista del caso sano",
                ("🧼 Ver RM sin máscara", "🧼 Ver RM con máscara (sin tumor)"),
                horizontal=True,
                key="vista_sano",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if vista == "🧼 Ver RM sin máscara":
                st.markdown(
                    "<h5 style='text-align:center'>RM original (sin tumor)</h5>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<h5 style='text-align:center'>RM con máscara (máscara vacía)</h5>",
                    unsafe_allow_html=True,
                )

            # En ambos casos se muestra la misma imagen, porque no hay tumor
            st.image(img_mri, use_column_width=False)

            # 🧼 Descripción clínico–analítica en inglés (sin tumor)
            st.markdown(
                """
                #### Clinical / data analyst interpretation (no visible tumor)

                - **Overall impression:** normal-appearing brain MRI for this slice, with
                  no focal mass, no clear edema pattern and preserved global symmetry.
                - **Segmentation point of view:** this is a **negative sample**; the
                  corresponding mask is empty, meaning no pixels are labelled as tumor.
                - **Why it matters for the model:** negative cases are crucial to reduce
                  false positives and to teach the network what healthy anatomy looks like.
                - **Expected behavior:** the model should assign low tumor probability to
                  all pixels in this image. Any high activation here would be a potential
                  false positive.
                """
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if tipo == "🔴 Con tumor":
            note_text = (
                "Nota: estos son ejemplos de cortes 2D con tumor. En la práctica se analizan "
                "volúmenes 3D y múltiples secuencias (T1, T2, FLAIR, contraste), junto con "
                "la historia clínica."
            )
        else:
            note_text = (
                "Nota: en estos casos no hay tumor en el corte mostrado. La 'máscara' es vacía, "
                "por lo que la RM con y sin máscara se ven iguales. Compararlos con los casos "
                "con tumor ayuda a entrenar y validar el modelo."
            )

        st.markdown(
            f"<p style='text-align:center; font-size:0.9rem;'>{note_text}</p>",
            unsafe_allow_html=True,
        )

def page_live_prediction():
    st.set_page_config(page_title="Brain MRI Tumor App",
                       page_icon="🧠",
                       layout="wide")
    
    API_URL = "http://localhost:5000/predict"  # URL de tu Flask
    
    def call_flask_model(api_url: str, pil_image: Image.Image):
        # Convertimos la PIL image a bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Enviamos como multipart/form-data
        files = {
            "image": ("mri.png", img_bytes, "image/png")
        }

        try:
            response = requests.post(api_url, files=files, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Error al llamar a la API: {e}")
            return None

        try:
            return response.json()
        except ValueError:
            st.error("La API no devolvió un JSON válido.")
            return None

    st.title("🧠 Detección de tumor en RM cerebral")

    uploaded_file = st.file_uploader(
        "Sube una imagen de RM cerebral (PNG/JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Mostrar imagen
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Imagen subida", use_column_width=True)

        if st.button("Analizar imagen"):
            with st.spinner("Enviando a la API de Flask y analizando..."):
                result = call_flask_model(API_URL, pil_img)

            if result is not None:
                has_cancer = result.get("has_cancer")
                prob = result.get("probability")
                label = result.get("label", "")

                if has_cancer:
                    st.error(f"Resultado: {label} (probabilidad: {prob:.2%})")
                else:
                    st.success(f"Resultado: {label} (probabilidad: {prob:.2%})")

                st.json(result)
   
def page_media():
    st.header("🎥 Visual demo and appointment")
    
    st.subheader("Demo video of the app / model")
    try:
        with open("video.mp4", "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception:
        st.info("Place a `video.mp4` file next to `app.py` or update the path in the code.")

    st.markdown(
        """
        In an integrated hospital environment, a similar interface could be embedded
        into the radiology workstation or electronic health record to:
        - Visualize AI-generated segmentations directly on the clinician's screen.
        - Provide structured summaries of tumor burden over time.
        - Suggest standardized follow-up intervals according to risk.
        For now, this demo focuses on showing the core concepts in an accessible way.
        """
    )

def page_contribute():
    st.header("🤝 Contribute and support patients")

    # Mensaje en verde
    st.success("June 8, 2023 – International Brain Tumor Day\n\n")

    st.markdown(
        """
        If you would like to support patients and research related to brain tumors
        and cancer in general, you can contribute through the  
        **Asociación Española Contra el Cáncer (AECC)**.
        
        The AECC is a non-profit organization that has been working for decades across
        Spain to **reduce the impact of cancer on society**. Its activity is based on
        three main pillars:
        
        - **Support for patients and families**: emotional, psychological and social care,
          guidance on practical aspects (work, benefits, legal issues) and accompaniment
          throughout the disease.
        - **Prevention and health education**: awareness campaigns about risk factors,
          promotion of healthy lifestyles and early detection programs.
        - **Funding of research**: competitive grants to research groups in oncology to
          improve diagnosis, treatment and survival.
        """
    )

    st.markdown(
        """
        In practice, the AECC acts through a network of local offices, hospitals and
        volunteer programs. They offer:
        
        - Free **psycho-oncological support** for patients and relatives.
        - **Social workers** who help manage administrative procedures and resources.
        - **Telephone and online helplines** to answer questions and provide guidance.
        - Collaboration with hospitals and research centers to drive **clinical and
          translational research**.
        """
    )

    # 🗓️ Bloque de cita con la asociación dentro
    st.subheader("📅 Follow-up appointment")

    cita = st.date_input(
        "Select a date for the follow-up visit",
        datetime.date.today(),
        key="contribute_date"
    )

    # Logo clicable de la AECC dentro del bloque de cita
    aecc_logo_url = "https://www.aecc.es/sites/default/files/styles/ps_xl/public/logo-aecc.png"
    st.markdown(
        f"""
        <a href="https://www.aecc.es/" target="_blank">
            <img src="{aecc_logo_url}" alt="AECC" width="260">
        </a>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        By clicking on the logo, you will be redirected to the official website of the
        Spanish Association Against Cancer, where you can:
        - Request information and support for you or your family.
        - Collaborate as a donor or volunteer.
        - Learn more about prevention and early detection.
        - Help fund cutting-edge cancer research projects.
        """
    )




def page_team():
    st.header("👥 Project team")

    st.markdown(
        """
        This work has been developed by a multidisciplinary team of students
        in Data Science and backend development.  
        
        Below you can see our profiles and GitHub links.
        """
    )

    st.markdown(
        """
        The project combines:
        - **Medical and domain knowledge**, to formulate clinically relevant questions.
        - **Machine learning and MLOps**, to design, train and deploy robust models.
        - **Data engineering**, to process raw DICOM images into analysis-ready tensors.
        - **Frontend and UX design**, to create interfaces that fit real clinical workflows.
        Effective AI in healthcare always requires this kind of cross-disciplinary collaboration.
        """
    )

def page_team():
    st.header("👥 Project team")

    st.markdown(
        """
        This work has been developed by a multidisciplinary team of students
        in Data Science and backend development.  
        
        Below you can see our profiles and GitHub links.
        """
    )

    team = [
        {
            "name": "Luna Pérez",
            "github": "https://github.com/LunaPerezT",
            "photo": "images/team/miembro1.jpg"
        },
        {
            "name": "Raquel Hernández",
            "github": "https://github.com/RaquelH18",
            "photo": "images/team/miembro2.jpg"
        },
        {
            "name": "María  Marín",
            "github": "https://github.com/mmarin3011-cloud",
            "photo": "images/team/miembro3.jpg"
        },
        {
            "name": "Fabián G. Martín",
            "github": "https://github.com/FabsGMartin",
            "photo": "images/team/miembro4.jpg"
        },
        {
            "name": "Miguel J. de la Torre",
            "github": "https://github.com/migueljdlt",
            "photo": "images/team/miembro5.jpg"
        },
        {
            "name": "Alejandro C.",
            "github": "https://github.com/alc98",
            "photo": "images/team/miembro6.jpg"
        },
    ]

    # Grid de 2 filas x 3 columnas, con GitHub justo debajo del nombre
    for row_start in range(0, len(team), 3):
        cols = st.columns(3)
        for col, member in zip(cols, team[row_start:row_start + 3]):
            with col:
                st.markdown("### " + member["name"])
                st.markdown(f"**GitHub:** [{member['github']}]({member['github']})")
                try:
                    img = Image.open(member["photo"])
                    st.image(img, use_column_width=True, caption=member["name"])
                except Exception:
                    st.info(f"Photo not found: `{member['photo']}`")

    st.info(
        """
        Although this is an academic project, any real-world deployment as a clinical
        tool would require collaboration with neuroradiologists, neurosurgeons,
        oncologists, medical physicists and hospital IT teams, as well as regulatory
        approval as a medical device.
        """
    )


def main():
    st.title("Brain MRI Tumor – Demo Streamlit")

    st.sidebar.header("Navigation")
    st.sidebar.caption("Choose a section to explore the project.")

def main():
    st.title("Brain MRI Tumor – Demo Streamlit")

    st.sidebar.header("Navigation")
    st.sidebar.caption("Choose a section to explore the project.")

    menu = [
        "🏠 Introduction",
        "🧬 Deep learning model",
        "📊 Database and charts",
        "🖼️ Example cases",
        "🔍 Live prediction",
        "🎥 Media and appointment",
        "🤝 Contribute",
        "👥 Team"
    ]

    choice = st.sidebar.radio("Select a page:", menu)

    if choice == "🏠 Introduction":
        page_intro()
    elif choice == "🧬 Deep learning model":
        page_model()
    elif choice == "📊 Database and charts":
        page_dataset()
    elif choice == "🖼️ Example cases":
        page_cases()
    elif choice == "🔍 Live prediction":
        page_live_prediction()
    elif choice == "🎥 Media and appointment":
        page_media()
    elif choice == "🤝 Contribute":
        page_contribute()
    elif choice == "👥 Team":
        page_team()

if __name__ == "__main__":
    main()









































































