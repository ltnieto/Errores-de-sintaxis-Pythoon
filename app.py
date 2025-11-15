
import streamlit as st
import joblib
import re

# --- Helper Functions ---
def remove_comments(code):
    # Remove single-line comments (starts with #, optionally preceded by whitespace)
    code = re.sub(r'#.*', '', code)
    # Remove multi-line comments (docstrings: """...""" or '''...''')
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    return code

def standardize_whitespace(code):
    # Remove leading/trailing whitespace from each line
    lines = code.split('
')
    cleaned_lines = [line.strip() for line in lines]
    # Remove empty lines
    cleaned_lines = [line for line in cleaned_lines if line]
    # Join lines back with a single newline, then remove extra spaces within lines
    code = '
'.join(cleaned_lines)
    # Replace multiple spaces with a single space, except for indentation
    code = re.sub(r'[ \t]+', ' ', code)
    return code.strip()

# --- Load Models and Preprocessors ---
# Using st.cache_resource to load models only once across Streamlit runs
@st.cache_resource
def load_models():
    try:
        model_syntax = joblib.load('model_syntax.joblib')
        model_structure = joblib.load('model_structure.joblib')
        tfidf_vectorizer_syntax = joblib.load('tfidf_vectorizer_syntax.joblib')
        tfidf_vectorizer_structure = joblib.load('tfidf_vectorizer_structure.joblib')
        return model_syntax, model_structure, tfidf_vectorizer_syntax, tfidf_vectorizer_structure
    except Exception as e:
        st.error(f"Error loading models or preprocessors: {e}. Please ensure joblib files are in the same directory as the app.py.")
        st.stop() # Stop the app if models cannot be loaded

model_syntax, model_structure, tfidf_vectorizer_syntax, tfidf_vectorizer_structure = load_models()

# --- Streamlit App Design ---
st.set_page_config(
    page_title="Code Snippet Analyzer",
    page_icon="✅", # A green checkmark emoji
    layout="centered"
)

st.title("✨ Analizador de Código")

st.markdown("--- ")

st.subheader("Ingresar Fragmento de Código")
user_code_input = st.text_area(
    "Pega tu código aquí para analizarlo:",
    "def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)"
)

if st.button("Analizar Código"): # Button to trigger analysis
    st.markdown("--- ")
    st.subheader("Resultados del Análisis")

    # Display original code for context
    st.write("**Código Ingresado:**")
    st.code(user_code_input, language='python')

    # Clean the user-provided code
    cleaned_user_code = standardize_whitespace(remove_comments(user_code_input))
    st.write("**Código Limpio (Preprocesado):**")
    st.code(cleaned_user_code, language='python')

    # --- Predict Syntax Error ---
    if cleaned_user_code: # Ensure there's text to analyze
        X_user_syntax = tfidf_vectorizer_syntax.transform([cleaned_user_code])
        syntax_prediction = model_syntax.predict(X_user_syntax)[0]

        prediction_message_syntax = "error de sintaxis" if syntax_prediction == 1 else "Código 10/10"
        st.write("**Detección de Errores de Sintaxis:**")
        if syntax_prediction == 1:
            st.error(prediction_message_syntax)
        else:
            st.success(prediction_message_syntax)

        # --- Predict Code Structure ---
        X_user_structure = tfidf_vectorizer_structure.transform([cleaned_user_code])
        structure_prediction = model_structure.predict(X_user_structure)[0]

        structure_labels = []
        if structure_prediction[0] == 1: structure_labels.append('Bucle (for/while)')
        if structure_prediction[1] == 1: structure_labels.append('Condicional (if/else)')
        if structure_prediction[2] == 1: structure_labels.append('Operación de Lista')

        st.write("**Clasificación de Estructura del Código:**")
        if structure_labels:
            st.info(f"Estructuras detectadas: {', '.join(structure_labels)}")
        else:
            st.info("No se detectaron estructuras de código específicas (bucle, condicional, operación de lista).")
    else:
        st.warning("Por favor, ingresa algún código para analizar.")

st.markdown("--- ")
st.info("Nota: Esta aplicación utiliza modelos pre-entrenados para demostración.")
