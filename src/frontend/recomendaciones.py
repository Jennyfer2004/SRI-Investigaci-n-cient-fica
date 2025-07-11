import streamlit as st
import requests
from deep_translator import GoogleTranslator

def mostrar_recomendaciones(traductor_es):
    """
    Muestra recomendaciones de preguntas al usuario en formato de botones.
    """
    st.write("**Preguntas que le  pueden interesar:**")
    
    recomendaciones = obtener_recomendaciones(st.session_state.user_id)
    if not recomendaciones:
        recomendaciones = [
            "¿Qué papers recientes hay sobre aplicaciones de transformers en biología?",
            "¿Cuáles son los últimos avances en modelos de lenguaje grandes para investigación médica?",
            "¿Puedes recomendarme papers sobre técnicas de fine-tuning para dominios específicos?",
            "¿Qué investigaciones recientes existen sobre ética en IA?",
            "¿Cuáles son los papers más citados sobre redes neuronales convolucionales?"
        ]
    else:
        try:
            recomendaciones = [traductor_es.translate(i) for i in recomendaciones]
        except Exception as e:
            recomendaciones = recomendaciones

    cols = st.columns(2)
    for i, pregunta in enumerate(recomendaciones[:4]):
        with cols[i % 2]:
            if st.button(pregunta, key=f"rec_{i}"):
                st.session_state.pregunta_actual = pregunta

def obtener_recomendaciones(user_id):
    """Obtiene recomendaciones personalizadas de preguntas desde el backend.
    Args:
        user_id(str): Identificador del usuario.
    Returns:
        list: Lista de preguntas recomendadas.
    """
    try:
        response = requests.post(
            "http://127.0.0.1:8000/recommendations",
            json={"user_id": user_id}
        )
        response.raise_for_status()
        return response.json()["recommendations"]
    except Exception as e:
        st.error(f"Error al obtener recomendaciones: {e}")
        return []