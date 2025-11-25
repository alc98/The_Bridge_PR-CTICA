from pathlib import Path
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    # Carpeta donde está este script
    base_path = Path(__file__).parent
    # Ruta al modelo dentro del proyecto
    model_path = base_path / "models" / "modelo_entrenado.pkl"

    # Opcional: debug
    # st.write(f"Intentando cargar modelo de: {model_path}")

    model = joblib.load(model_path)
    return model



# Interfaz

st.title("📊 Sentiment Classifier for Reviews")
st.write("Binary model: **0 = negative**, **1 = positive**")

st.info("Note: The review **must be written in English**. Predictions may be unreliable for other languages.")

review_text = st.text_area(
    "Type the customer's review in English:",
    height=150,
    placeholder="E.g.: The service was very slow and the food arrived cold..."
)

if st.button("Classify sentiment"):
    if not review_text.strip():
        st.warning("Please type some text before classifying.")
    else:
        
        prediction = model.predict([review_text])[0]

        
        
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([review_text])[0]

        if prediction == 1:
            label = "👍 Positive (1)"
        else:
            label = "👎 Negative (0)"

        st.markdown(f"### Result: {label}")

        if probabilities is not None:
            prob_0 = float(probabilities[0])
            prob_1 = float(probabilities[1])
            st.write("Estimated probabilities:")
            st.write(f"- 0 (negative): **{prob_0:.2%}**")
            st.write(f"- 1 (positive): **{prob_1:.2%}**")

            
            st.progress(prob_1 if prediction == 1 else prob_0)



