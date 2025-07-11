import streamlit as st
import requests
import re 
from deep_translator import GoogleTranslator
from recomendaciones import mostrar_recomendaciones,obtener_recomendaciones
traductor_es = GoogleTranslator(source='auto', target='es')
traductor_en = GoogleTranslator(source='auto', target='en')
import recomendaciones

st.title("Chatbot RAG de investigación científica con Streamlit")

# Inicialización del estado de la sesión
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = 0
if "user_id" not in st.session_state:
    st.session_state.user_id = str(hash(st.experimental_user.email)) 
if f"chat_{st.session_state.current_chat_id}" not in st.session_state.chats:
    st.session_state.chats[f"chat_{st.session_state.current_chat_id}"] = {
        "messages": [],
        "show_recommendations": True
    }



def nuevo_chat():
    """Crea un nuevo chat en la sesión actual.
    Incrementa el ID del chat actual y añade una nueva entrada vacía al diccionario de chats.
    """
    st.session_state.current_chat_id += 1
    st.session_state.chats[f"chat_{st.session_state.current_chat_id}"] = {
        "messages": [],
        "show_recommendations": True
    }

# Barra lateral izquierda con lista de chats
with st.sidebar:
    st.write("**Chats**")
    if st.button("+ Nuevo Chat"):
        nuevo_chat()
    for chat_id in st.session_state.chats:
        if st.button(f"Chat {chat_id}"):
            st.session_state.current_chat_id = int(chat_id.split("_")[1])

# Chat actual
current_chat = st.session_state.chats[f"chat_{st.session_state.current_chat_id}"]
messages = current_chat["messages"]
show_recommendations = current_chat["show_recommendations"]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obtener y mostrar recomendaciones personalizadas
if True:
    st.write("**Preguntas que le  pueden interesar:**")
    
    recomendaciones = obtener_recomendaciones(st.session_state.user_id)

    if recomendaciones:
        try:
            recomendaciones = [traductor_es.translate(i) for i in recomendaciones]
        except Exception as e:
            recomendaciones = recomendaciones

    else:
        recomendaciones = [
            "¿Qué papers recientes hay sobre aplicaciones de transformers en biología?",
            "¿Cuáles son los últimos avances en modelos de lenguaje grandes para investigación médica?",
            "¿Puedes recomendarme papers sobre técnicas de fine-tuning para dominios específicos?",
            "¿Qué investigaciones recientes existen sobre ética en IA?",
            "¿Cuáles son los papers más citados sobre redes neuronales convolucionales?"
        ]
    cols = st.columns(2)
    for i, pregunta in enumerate(recomendaciones[:4]): 
        with cols[i % 2]:
            if st.button(pregunta):
                st.session_state.pregunta_actual = pregunta

user_input = st.chat_input("Haz tu pregunta aquí")

if "pregunta_actual" in st.session_state and not user_input:
    user_input = st.session_state.pregunta_actual
    del st.session_state.pregunta_actual

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    messages.append({"role": "user", "content": user_input})
    current_chat["show_recommendations"] = False

    pregunta_en = traductor_en.translate(user_input) if user_input else ""

    try:
        response = requests.post(
            "http://127.0.0.1:8000/query",
            json={
                "question": pregunta_en,
                "user_id": st.session_state.user_id
            }
        )
        response.raise_for_status()
        respuesta = response.json()["respuesta"]
        metadatos = response.json()["recomendaciones"]
        match = re.search(r"Contenido:\s*(.*)", respuesta, re.DOTALL)
        if match:
            respuesta=match.group(1)
        
    except Exception as e:
        respuesta = f"⚠️ Error al conectar con el servidor: {e}"
        metadatos = []
        
    respuesta_es = traductor_es.translate(respuesta) if not respuesta.startswith("⚠️") else respuesta

    with st.chat_message("assistant"):
        st.markdown(respuesta_es)
        
        if metadatos:
            st.markdown("---")
            st.subheader("📄 Papers de referencia:")
            for paper in metadatos:
                with st.expander(f"**{paper['titulo']}**"):
                    st.write(f"**Autores:** {paper['autores']}")
                    st.write(f"**Publicado:** {paper['publicado']}")
                    st.write(f"**Idioma:** {paper['idioma']}")
                    st.markdown(f"**DOI:** [{paper['doi']}](https://doi.org/{paper['doi']})")
  
                    
                    if paper.get('url_pdf'):
                        st.markdown(f"[📄 Ver PDF]({paper['url_pdf']})")
                    if paper.get('url_landing'):
                        st.markdown(f"[🌐 Página oficial]({paper['url_landing']})")
    messages.append({"role": "assistant", "content": respuesta_es})

    mostrar_recomendaciones(traductor_es)