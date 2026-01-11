"""
Sistema de Detección de Anomalías en Termografías de Mama
=========================================================

Aplicación Streamlit para clasificación de imágenes termográficas
utilizando ResNet50 + SVM.

Proyecto de Tesis - Maestría en Inteligencia Artificial Aplicada
Universidad Técnica Particular de Loja
Autor: Julio César Luna Díaz
Director: Luis Rodrigo Barba Guamán
Año: 2025
"""

import streamlit as st
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.predictor import ThermographyPredictor

# Configuración de la página
st.set_page_config(
    page_title="Detección de Anomalías en Termografías de Mama",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    /* Estilo del header principal */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666;
    }

    /* Estilos para resultados */
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }

    .result-healthy {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }

    .result-anomaly {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-message {
        font-size: 1rem;
        color: #333;
    }

    .confidence-text {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* Disclaimer */
    .disclaimer {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }

    /* Métricas */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f4e79;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }

    /* Sidebar mejorado */
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .sidebar-title {
        font-weight: 600;
        color: #1f4e79;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """
    Carga el predictor con caché para evitar recargas.
    """
    model_path = ROOT_DIR / "results" / "models" / "resnet_svm.joblib"
    return ThermographyPredictor(str(model_path))


def render_header():
    """Renderiza el encabezado principal."""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🏥 Sistema de Detección de Anomalías en Termografías de Mama</div>
        <div class="subtitle">
            Análisis inteligente de imágenes termográficas para detección temprana
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renderiza la barra lateral con información y controles."""
    with st.sidebar:
        st.markdown("### 📤 Cargar Imagen")
        uploaded_file = st.file_uploader(
            "Seleccione una imagen termográfica",
            type=["png", "jpg", "jpeg"],
            help="Formatos soportados: PNG, JPG, JPEG"
        )

        st.markdown("---")

        # Acerca de
        st.markdown("### ℹ️ Acerca de")
        st.markdown("""
        **Proyecto de Tesis**
        Maestría en Inteligencia Artificial Aplicada

        **Universidad**
        Universidad Técnica Particular de Loja

        **Autor**
        Julio César Luna Díaz

        **Director**
        Luis Rodrigo Barba Guamán

        **Año:** 2025
        """)

        st.markdown("---")

        # Información del modelo
        st.markdown("### 📊 Información del Modelo")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "90.24%")
        with col2:
            st.metric("AUC-ROC", "0.989")

        st.markdown("""
        **Arquitectura:**
        - **Extractor:** ResNet50 (ImageNet)
        - **Clasificador:** SVM (RBF)

        **Dataset:** DMR-IR
        """)

        st.markdown("---")

        # Métricas detalladas
        with st.expander("Ver métricas detalladas"):
            st.markdown("""
            | Métrica | Valor |
            |---------|-------|
            | Accuracy | 90.24% |
            | Precision | 85.71% |
            | Recall | 85.71% |
            | F1-Score | 85.71% |
            | Specificity | 92.59% |
            | AUC-ROC | 0.9894 |
            """)

    return uploaded_file


def render_disclaimer():
    """Renderiza el aviso médico importante."""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚕️ Aviso importante:</strong> Este sistema es una herramienta de apoyo al diagnóstico
        y <strong>NO reemplaza</strong> la evaluación médica profesional. Ante cualquier duda,
        consulte siempre con un especialista en salud.
    </div>
    """, unsafe_allow_html=True)


def render_result(result: dict):
    """Renderiza el resultado de la predicción."""
    prediction = result['prediction']
    confidence = result['confidence']

    if prediction == 0:  # Saludable
        st.markdown(f"""
        <div class="result-box result-healthy">
            <div class="result-title">✅ Resultado: Saludable</div>
            <div class="result-message">
                No se detectaron anomalías térmicas significativas en la imagen analizada.
            </div>
            <div class="confidence-text">
                Confianza: {confidence * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:  # Anomalía
        st.markdown(f"""
        <div class="result-box result-anomaly">
            <div class="result-title">⚠️ Resultado: Anomalía Detectada</div>
            <div class="result-message">
                Se han detectado patrones térmicos que podrían indicar una anomalía.
                Se recomienda consultar con un especialista médico.
            </div>
            <div class="confidence-text">
                Confianza: {confidence * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Barra de progreso de confianza
    st.markdown("#### Nivel de Confianza")
    st.progress(confidence)

    # Probabilidades por clase
    st.markdown("#### Probabilidades por Clase")
    col1, col2 = st.columns(2)
    with col1:
        prob_healthy = result['probabilities']['saludable']
        st.metric("Saludable", f"{prob_healthy * 100:.1f}%")
    with col2:
        prob_anomaly = result['probabilities']['anomalia']
        st.metric("Anomalía", f"{prob_anomaly * 100:.1f}%")


def render_images(result: dict, show_clahe: bool = True):
    """Renderiza las imágenes original y preprocesada."""
    if show_clahe:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Imagen Original")
            st.image(result['original_image'], use_container_width=True)
        with col2:
            st.markdown("##### Imagen con CLAHE")
            st.image(result['clahe_image'], use_container_width=True)
    else:
        st.markdown("##### Imagen Cargada")
        st.image(result['original_image'], use_container_width=True)


def render_footer():
    """Renderiza el pie de página."""
    st.markdown("""
    <div class="footer">
        <p>
            <strong>Universidad Técnica Particular de Loja</strong> |
            Maestría en Inteligencia Artificial Aplicada | 2025
        </p>
        <p style="font-size: 0.85rem;">
            Desarrollado como parte de la tesis: "Evaluación de modelos de redes neuronales
            sobre imágenes termográficas para la detección del cáncer de mama"
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Función principal de la aplicación."""
    # Renderizar encabezado
    render_header()

    # Renderizar barra lateral y obtener archivo cargado
    uploaded_file = render_sidebar()

    # Descripción del sistema
    st.markdown("""
    Este sistema utiliza **inteligencia artificial** para analizar imágenes termográficas
    y detectar posibles anomalías que podrían indicar la presencia de cáncer de mama.
    El modelo fue entrenado con el dataset **DMR-IR** y alcanza una precisión del **90.24%**.
    """)

    # Disclaimer
    render_disclaimer()

    st.markdown("---")

    # Área principal
    if uploaded_file is None:
        # Instrucciones cuando no hay imagen
        st.markdown("### 📋 Instrucciones de Uso")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **1. Cargar Imagen**

            Use el panel lateral para cargar una imagen termográfica en formato PNG, JPG o JPEG.
            """)

        with col2:
            st.markdown("""
            **2. Visualizar**

            Revise la imagen cargada y la versión preprocesada con mejora de contraste (CLAHE).
            """)

        with col3:
            st.markdown("""
            **3. Analizar**

            Presione el botón "Analizar" para obtener la clasificación y el nivel de confianza.
            """)

        st.info("👈 Cargue una imagen termográfica desde el panel lateral para comenzar el análisis.")

    else:
        # Imagen cargada
        try:
            # Cargar el predictor
            with st.spinner("Cargando modelo..."):
                predictor = load_predictor()

            # Leer bytes de la imagen
            image_bytes = uploaded_file.read()

            # Vista previa y análisis
            col_img, col_result = st.columns([1, 1])

            with col_img:
                st.markdown("### 🖼️ Imagen Cargada")
                st.image(image_bytes, use_container_width=True)

            with col_result:
                st.markdown("### 🔍 Análisis")

                # Botón de análisis
                if st.button("🔬 Analizar Imagen", type="primary", use_container_width=True):
                    with st.spinner("Procesando imagen..."):
                        # Realizar predicción
                        result = predictor.predict(image_bytes)

                    # Guardar resultado en session state
                    st.session_state['result'] = result

                # Mostrar resultado si existe
                if 'result' in st.session_state:
                    render_result(st.session_state['result'])

            # Mostrar imágenes preprocesadas si hay resultado
            if 'result' in st.session_state:
                st.markdown("---")
                st.markdown("### 📸 Comparación de Preprocesamiento")

                show_clahe = st.checkbox("Mostrar imagen con CLAHE aplicado", value=True)
                render_images(st.session_state['result'], show_clahe)

        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")
            st.info("Asegúrese de cargar una imagen válida en formato PNG, JPG o JPEG.")

    # Footer
    st.markdown("---")
    render_footer()


if __name__ == "__main__":
    main()
