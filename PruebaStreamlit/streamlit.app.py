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

def page_model():
    st.header("🧬 Deep learning model")
    st.markdown(
        """
        In a real hospital workflow, such a model would typically act as a
        **decision-support tool** or “second reader”. It can:
        - Highlight suspicious regions that deserve closer inspection.
        - Provide quantitative measurements (e.g. tumor volume).
        - Help standardize reports across radiologists.
        Final responsibility for diagnosis and treatment decisions always remains
        with the clinical team.
        """
    )

    st.markdown("## General architecture")
    st.markdown(
        """
        Our medical AI system is based on a **deep learning model** that operates on
        brain MRI slices.

        At a high level, the pipeline is:

        1. **Input**: MRI image (normalized and resized).
        2. **Neural network** (e.g. U-Net or CNN):
           - Extracts visual patterns (edges, textures, hyperintense regions...).
           - Learns to distinguish between healthy tissue and tumor tissue.
        3. **Output**:
           - A **class prediction**: tumor / no tumor.
           - Optionally, a **segmentation mask** highlighting tumor pixels.
        """
    )

    st.markdown(
        """
        Although this demo focuses on 2D slices, many research systems work with:
        - **3D convolutions**, which exploit volumetric context across slices.
        - **Multi-sequence input** (T1, T1+contrast, T2, FLAIR) stacked as channels.
        - **Multimodal fusion**, combining imaging with clinical variables
          (age, performance status, molecular markers) or even genomics.
        This richer input can improve performance for tasks such as grading or prognosis.
        """
    )

    st.markdown("## Training (summary)")
    st.markdown(
        """
        - **Data**: MRI dataset with tumor annotations.
        - **Labels**:
          - For classification: `0` = no tumor, `1` = tumor.
          - For segmentation: masks where each pixel indicates tumor/no tumor.
        - **Procedure**:
          - Split into *train / validation / test*.
          - Train for several epochs minimizing a loss function
            (for example, *Binary Cross-Entropy* for classification or
            *Dice loss* for segmentation).
        - **Typical metrics**:
          - Classification: accuracy, F1, sensitivity, specificity.
          - Segmentation: Dice coefficient, IoU.
        """
    )

    st.markdown("## Data preprocessing and quality control")
    st.markdown(
        """
        Before training any medical imaging model, a robust preprocessing pipeline is essential:
        - **Skull stripping** to remove non-brain tissue and reduce noise.
        - **Intensity normalization** per scan to mitigate scanner- or protocol-related variability.
        - **Spatial registration** to a common template when combining data from multiple patients.
        - **Resampling to isotropic voxels** so that physical distances are comparable.
        - **Data augmentation** (rotations, flips, elastic deformations, mild intensity shifts)
          to improve generalization and simulate real-world acquisition variability.
        A careful visual QC (quality control) step is usually performed with radiologists
        to exclude corrupted or mislabeled scans.
        """
    )

    st.markdown("## Evaluation and clinical interpretation")
    st.markdown(
        """
        Beyond global metrics, clinicians and data scientists typically:
        - Inspect **ROC and precision-recall curves** to select thresholds that balance
          sensitivity (avoiding missed tumors) and specificity (avoiding unnecessary alarms).
        - Use **calibration curves** to verify that predicted probabilities correspond
          to actual risk, which is crucial when communicating risk to patients.
        - Analyze **confusion matrices** stratified by subgroups (age, sex, scanner type,
          tumor location) to detect potential bias.
        - Compare performance with human experts in **reader studies** and investigate
          cases where the model disagrees with the radiologist.
        - Perform **external validation** on data from other hospitals to test
          generalization beyond the training cohort.
        """
    )

    st.markdown("## Integration with Flask")
    st.info(
        """
        The model is deployed inside a **Flask API**:

        - The Flask app exposes an HTTP endpoint (for example, `/predict`).
        - Streamlit sends the MRI image to the endpoint in base64 format.
        - Flask runs the deep learning model and returns:
          - whether there is a tumor (`has_tumor`)
          - the probability (`probability`)
          - optionally, a mask (`mask_base64`).

        This separation allows us to:
        - Scale the model independently (GPU/CPU).
        - Use Streamlit only as a lightweight visual interface.
        """
    )

    st.markdown(
        """
        In a production setting, this architecture would be complemented with:
        - **Authentication and audit logs** to track who requested each prediction.
        - **Versioning** of models and training datasets to ensure reproducibility.
        - **Monitoring** of latency, error rates and data drift to detect when
          the model may need to be re-evaluated or retrained.
        - Integration with hospital systems (PACS/RIS) using standards such as DICOM and HL7/FHIR.
        """
    )

    st.markdown("## Limitations and responsible use")
    st.warning(
        """
        This application is a **proof of concept** (PoC):

        - It does not replace the judgment of a medical professional.
        - Predictions may contain errors.
        - Any real clinical use must undergo rigorous validation.
        """
    )

    st.info(
        """
        Even models that perform well in retrospective studies can fail once deployed
        if the patient population, scanners or imaging protocols change over time.
        Continuous surveillance, periodic re-validation and collaboration between
        data scientists, clinicians and MLOps engineers are essential for safe,
        responsible AI in healthcare.
        """
    )



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

        # 👉 Primero SIN tumor, luego CON tumor
        tipo = st.radio(
            "Selecciona tipo de caso",
            ("🟢 Sin tumor", "🔴 Con tumor"),
            horizontal=True,
            index=0,   # por defecto: Sin tumor
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

        else:
            # ejemplo sin tumor: una única RM -> generamos máscara negra
            img_mri = Image.open(current_path).convert("RGB")
            w, h = img_mri.size
            img_mask = Image.new("L", (w, h), color=0)  # negro

            # Solo 2 imágenes para "limpiar" la vista
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(
                    "<h5 style='text-align:center'>RM original (sin tumor)</h5>",
                    unsafe_allow_html=True,
                )
                st.image(img_mri, use_column_width=True)

            with c2:
                st.markdown(
                    "<h5 style='text-align:center'>Máscara (sin tumor)</h5>",
                    unsafe_allow_html=True,
                )
                st.image(img_mask, use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if tipo == "🔴 Con tumor":
            note_text = (
                "Nota: estos son ejemplos de cortes 2D con tumor. En la práctica se analizan "
                "volúmenes 3D y múltiples secuencias (T1, T2, FLAIR, contraste), junto con "
                "la historia clínica."
            )
        else:
            note_text = (
                "Nota: en estos casos no hay tumor en el corte mostrado y la máscara permanece "
                "vacía. Compararlos con los casos con tumor ayuda a entrenar y validar el modelo."
            )

        st.markdown(
            f"<p style='text-align:center; font-size:0.9rem;'>{note_text}</p>",
            unsafe_allow_html=True,
        )


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







































