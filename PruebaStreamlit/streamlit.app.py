import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import datetime
import requests
import base64
import io
import numpy as np


# Page config & globals

try:
    logo = Image.open("logo.png")
    page_icon = logo
except Exception:
    logo = None
    page_icon = "🧠"

st.set_page_config(
    page_title="Brain MRI Tumor – Deep Learning Demo",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_data
def load_data(path: str = "data.csv") -> pd.DataFrame:
    """Load cohort CSV. Returns empty DataFrame if not found."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame()
    return df

df = load_data()


GENDER_COL = "Gender"
TUMOR_COL = "Tumor"  #0 = no tumor, 1 = tumor



# API client utilities (Flask model)

def call_flask_model(api_url: str, pil_image: Image.Image):
    """Send an MRI slice to the Flask API and return the JSON response."""
    pil_image = pil_image.convert("RGB")

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    url = api_url.rstrip("/") + "/predict"

    resp = requests.post(
        url,
        json={"image_base64": img_b64},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def decode_mask_from_b64(mask_b64: str) -> np.ndarray:
    """Decode a PNG mask from base64 to a NumPy array."""
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes))
    return np.array(mask_img)



# PAGE 1 – TECHNICAL OVERVIEW

def page_intro():
    st.header("🧠 Brain Tumor Detection and Segmentation")

    if logo is not None:
        st.image(logo, width=120)

    st.markdown("## Clinical and technical overview")

    st.info(
        """
        This demo focuses on **automatic brain tumor analysis** from structural MRI,
        with a particular emphasis on **low-grade gliomas (LGG)** and related lesions.

        From a **machine learning** perspective, the problem can be formulated as:
        - **Binary classification**: *tumor present* vs. *no visible tumor*.
        - **Semantic segmentation**: pixel-wise delineation of tumor tissue.
        """
    )

    st.markdown(
        """
        ### Pipeline (high-level)

        1. **Input data**  
           - 2D MRI slices (e.g., T1, T2, FLAIR) preprocessed and resampled.
           - Basic metadata such as gender, age, and tumor label.

        2. **Deep learning models**  
           - **Classification**: convolutional neural networks (CNNs) that output a probability
             of tumor presence for a given slice or volume.
           - **Segmentation**: encoder–decoder architectures such as **U-Net**, predicting
             a binary mask over the brain.

        3. **Outputs**  
           - A **probability score** (e.g., `p(tumor)`).
           - Optional **segmentation masks** highlighting the tumor region.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        In this Streamlit app you can explore:

        - **Cohort statistics** and class balance (dataset page).
        - **Example negative and positive cases** with masks.
        - A **live inference demo** calling a Flask API that wraps
          the deep learning model.
        """
    )



# PAGE 2 – DATASET & EDA 

def page_dataset():
    st.header("📊 Dataset and Exploratory Analysis")

    if df.empty:
        st.error(
            "No `data.csv` file was found. Place it next to `app.py` and reload "
            "the app, or update the path in `load_data()`."
        )
        return

    st.caption(f"Rows: {df.shape[0]} · Columns: {df.shape[1]}")

    tab_table, tab_charts, tab_schema = st.tabs(
        ["📄 Data table", "📈 Charts", "🧾 Schema / dtypes"]
    )

    with tab_table:
        st.subheader("Raw dataset")
        st.dataframe(df)

    with tab_schema:
        st.subheader("Column summary")
        st.write(df.dtypes.to_frame("dtype"))

    with tab_charts:
        # Gender distribution
        if GENDER_COL in df.columns:
            st.markdown("### Gender distribution")

            df_count = df.groupby(GENDER_COL).size().reset_index(name="count")

            fig_pie = px.pie(
                df_count,
                values="count",
                names=GENDER_COL,
                title="Patients by gender",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info(f"Column `{GENDER_COL}` not found in the CSV.")

        # Tumor statistics if tumor column exists
        if TUMOR_COL in df.columns:
            st.markdown("### Tumor probability by gender")

            if GENDER_COL in df.columns:
                df_avg = (
                    df.groupby(GENDER_COL)[TUMOR_COL]
                    .mean()
                    .reset_index(name="Tumor_Prob")
                )

                fig_bar = px.bar(
                    df_avg,
                    x=GENDER_COL,
                    y="Tumor_Prob",
                    title="Average tumor probability by gender",
                    labels={"Tumor_Prob": "Tumor probability"},
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("### Query by gender")
                genders = df[GENDER_COL].dropna().unique().tolist()
                sel_gender = st.selectbox("Select gender", genders)

                prob_sel = df_avg.loc[
                    df_avg[GENDER_COL] == sel_gender, "Tumor_Prob"
                ].values
                if len(prob_sel) > 0:
                    st.success(
                        f"Estimated mean tumor probability for **{sel_gender}**: "
                        f"**{prob_sel[0] * 100:.2f}%**"
                    )

            st.markdown("### Global class distribution (tumor vs no tumor)")

            class_counts = df[TUMOR_COL].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]

            fig_bool = px.bar(
                class_counts,
                x="Class",
                y="Count",
                title="Number of patients per class (0 = no tumor, 1 = tumor)",
                text="Count",
            )
            st.plotly_chart(fig_bool, use_container_width=True)
        else:
            st.info(
                f"Column `{TUMOR_COL}` not found. Cannot compute tumor-related "
                "probabilities or class distribution."
            )

        # Generic numeric-column exploration
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            st.markdown("### Univariate distribution (numeric feature)")
            num_col = st.selectbox(
                "Select a numeric column", numeric_cols, key="numeric_eda_select"
            )
            fig_hist = px.histogram(
                df,
                x=num_col,
                nbins=30,
                title=f"Distribution of `{num_col}`",
            )
            st.plotly_chart(fig_hist, use_container_width=True)



# PAGE 3 – MODEL EXPLANATION

def page_model():
    st.header("🧬 Deep Learning Model")

    st.markdown("## Architecture")

    st.markdown(
        """
        The underlying model is a **convolutional neural network** trained on brain MRI:

        - **Backbone**: 2D or 3D CNN with multiple convolution–ReLU–pooling blocks.
        - **Segmentation head** (optional): U-Net–style decoder with skip connections
          for high-resolution predictions.
        - **Classification head**:
          - Global average pooling.
          - Fully connected layers.
          - Sigmoid / softmax output for tumor probability.

        The input images are typically:
        - Intensity-normalized.
        - Resampled to a fixed in-plane resolution.
        - Cropped or padded to a fixed spatial size.
        """
    )

    st.markdown("## Training setup")

    st.markdown(
        """
        **Data and labels**

        - MRI slices or volumes with **tumor annotations**.
        - Classification label: `0` = no tumor, `1` = tumor.
        - Segmentation label: binary mask per voxel / pixel.

        **Training procedure**

        - Split into *train / validation / test*.
        - Data augmentation (flips, rotations, small intensity shifts).
        - Optimizer: Adam or similar, with a learning-rate schedule.
        - Early stopping based on validation loss or Dice score.

        **Loss functions**

        - Classification:
          - Binary Cross-Entropy (BCE) or focal loss.
        - Segmentation:
          - Dice loss, BCE + Dice, or variants (Tversky, focal Tversky).
        """
    )

    st.markdown("## Evaluation metrics")

    st.markdown(
        """
        - **Classification**
          - Accuracy
          - Sensitivity (recall, true positive rate)
          - Specificity (true negative rate)
          - ROC AUC, PR AUC

        - **Segmentation**
          - Dice coefficient
          - Intersection over Union (IoU)
          - Precision / recall at pixel level
        """
    )

    st.markdown("## Deployment via Flask API")

    st.info(
        """
        The trained model is wrapped inside a **Flask API**:

        - Streamlit sends the MRI slice encoded as **base64 PNG** to an endpoint
          (e.g. `/predict`).
        - Flask runs the deep learning model and returns:
          - `has_tumor` (boolean)
          - `probability` (float in [0, 1])
          - `mask_base64` (optional, PNG-encoded segmentation mask)

        This design decouples:
        - **Model serving** (Flask, potentially with GPU acceleration).
        - **User interface** (this Streamlit dashboard).
        """
    )

    st.warning(
        """
        This application is a **proof of concept** and is **not** intended for
        clinical decision making. Any real-world deployment must undergo
        rigorous validation, regulatory approval, and monitoring.
        """
    )


# -------------------------------------------------------------------
# PAGE 4 – EXAMPLE CASES WITH MASKS
# -------------------------------------------------------------------
def page_cases():
    st.header("🖼️ Example Cases: Negative vs Positive")

    st.markdown(
        "This section illustrates a **negative case** (no visible tumor) and a "
        "**positive case** (tumor present), along with their segmentation masks."
    )

    neg_img_path = "images/negative_case_mri.png"
    neg_mask_path = "images/negative_case_mask.png"
    pos_img_path = "images/positive_case_mri.png"
    pos_mask_path = "images/positive_case_mask.png"

    st.markdown("### Negative case (no tumor)")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("MRI – negative case")
        try:
            neg_img = Image.open(neg_img_path)
            st.image(neg_img, use_column_width=True)
        except Exception:
            st.info(f"Place the negative-case MRI at `{neg_img_path}`.")

    with col2:
        st.caption("Mask – negative case (no tumor region)")
        try:
            neg_mask = Image.open(neg_mask_path)
            st.image(neg_mask, use_column_width=True)
        except Exception:
            st.info(f"Place the negative-case mask at `{neg_mask_path}`.")

    st.markdown("---")
    st.markdown("### Positive case (with tumor)")
    col3, col4 = st.columns(2)

    with col3:
        st.caption("MRI – positive case")
        try:
            pos_img = Image.open(pos_img_path)
            st.image(pos_img, use_column_width=True)
        except Exception:
            st.info(f"Place the positive-case MRI at `{pos_img_path}`.")

    with col4:
        st.caption("Mask – positive case (tumor highlighted)")
        try:
            pos_mask = Image.open(pos_mask_path)
            st.image(pos_mask, use_column_width=True)
        except Exception:
            st.info(f"Place the positive-case mask at `{pos_mask_path}`.")



# PAGE 5 – LIVE PREDICTION (STREAMLIT -> FLASK)

def page_live_prediction():
    st.header("🔍 Live Inference via Flask Model")

    st.markdown(
        """
        Upload a brain MRI slice and the app will query the **deep learning model**
        served by a Flask API to predict whether a tumor is present.
        """
    )

    st.sidebar.markdown("### ⚙️ Flask API configuration")
    api_url = st.sidebar.text_input("Base API URL", "http://localhost:8000")

    uploaded_file = st.file_uploader(
        "Upload a brain MRI slice (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded MRI", use_column_width=True)

        if st.button("Run inference"):
            with st.spinner("Querying Flask model..."):
                try:
                    response = call_flask_model(api_url, pil_img)
                except Exception as e:
                    st.error(f"Error calling the API: {e}")
                    return

            st.markdown("### Model output")

            has_tumor = response.get("has_tumor", None)
            prob = response.get("probability", None)

            if has_tumor is None or prob is None:
                st.error(
                    "The API response does not contain the expected keys "
                    "(`has_tumor`, `probability`). Please adapt the code to "
                    "your actual API schema."
                )
            else:
                diagnosis = "TUMOR DETECTED" if has_tumor else "NO TUMOR DETECTED"
                color = "🔴" if has_tumor else "🟢"

                st.metric(
                    label="Model diagnosis",
                    value=f"{color} {diagnosis}",
                )
                st.metric(
                    label="Tumor probability",
                    value=f"{prob * 100:.2f} %",
                )

            mask_b64 = response.get("mask_base64", None)
            if mask_b64:
                st.markdown("### Segmentation mask (optional)")
                try:
                    mask_arr = decode_mask_from_b64(mask_b64)
                    st.image(
                        mask_arr,
                        caption="Segmentation mask predicted by the model",
                        use_column_width=True,
                    )
                except Exception:
                    st.info("The returned mask could not be decoded.")



# PAGE 6 – MEDIA & APPOINTMENT

def page_media():
    st.header("🎥 Visual Demo and Appointment Simulation")

    st.subheader("Sample images")

    try:
        img_local = Image.open("image.png")
        st.image(img_local, caption="Local sample image", use_column_width=True)
    except Exception:
        st.info(
            "Place an image named `image.png` next to `app.py` or update the path "
            "in the code."
        )

    st.image(
        "https://picsum.photos/1280",
        caption="Sample image from URL",
        use_column_width=True,
    )

    st.subheader("Demo video of the app / model")
    try:
        with open("video.mp4", "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except Exception:
        st.info(
            "Place `video.mp4` next to `app.py` or update the path in the code."
        )

    st.subheader("📅 Follow-up appointment (mock)")
    appointment = st.date_input(
        "Select a date for a follow-up appointment", datetime.date.today()
    )
    st.success(f"Selected date: {appointment.strftime('%Y-%m-%d')}")



# PAGE 7 – TEAM

def page_team():
    st.header("👥 Project Team")

    st.markdown(
        """
        This project was developed by a **multidisciplinary team** of students in
        data science and backend engineering.

        Below you can find our profiles and GitHub links.
        """
    )

    team = [
        {
            "name": "Member 1",
            "role": "ML Engineer",
            "github": "https://github.com/user1",
            "photo": "images/team/member1.jpg",
        },
        {
            "name": "Member 2",
            "role": "Backend Developer",
            "github": "https://github.com/user2",
            "photo": "images/team/member2.jpg",
        },
        {
            "name": "Member 3",
            "role": "Data Scientist",
            "github": "https://github.com/user3",
            "photo": "images/team/member3.jpg",
        },
        {
            "name": "Member 4",
            "role": "MLOps & Cloud",
            "github": "https://github.com/user4",
            "photo": "images/team/member4.jpg",
        },
        {
            "name": "Member 5",
            "role": "Frontend & UX",
            "github": "https://github.com/user5",
            "photo": "images/team/member5.jpg",
        },
        {
            "name": "Member 6",
            "role": "Data Engineer",
            "github": "https://github.com/user6",
            "photo": "images/team/member6.jpg",
        },
    ]

    for row_start in range(0, len(team), 3):
        cols = st.columns(3)
        for col, member in zip(cols, team[row_start : row_start + 3]):
            with col:
                st.markdown("### " + member["name"])
                try:
                    img = Image.open(member["photo"])
                    st.image(img, use_column_width=True, caption=member["role"])
                except Exception:
                    st.info(f"Photo not found: `{member['photo']}`")

                st.markdown(f"[🌐 GitHub]({member['github']})", unsafe_allow_html=True)



# MAIN NAVIGATION

def main():
    st.title("Brain MRI Tumor – Deep Learning Demo")

    st.sidebar.header("Navigation")
    st.sidebar.caption("Choose a section to explore the project.")

    menu = [
        "🏠 Technical overview",
        "📊 Dataset & exploratory analysis",
        "🧬 Deep learning model",
        "🖼️ Example cases",
        "🔍 Live prediction",
        "🎥 Media & appointment",
        "👥 Team",
    ]

    choice = st.sidebar.radio("Select a page:", menu)

    if choice == "🏠 Technical overview":
        page_intro()
    elif choice == "📊 Dataset & exploratory analysis":
        page_dataset()
    elif choice == "🧬 Deep learning model":
        page_model()
    elif choice == "🖼️ Example cases":
        page_cases()
    elif choice == "🔍 Live prediction":
        page_live_prediction()
    elif choice == "🎥 Media & appointment":
        page_media()
    elif choice == "👥 Team":
        page_team()


if __name__ == "__main__":
    main()
