import os
# import json
# import requests
# from typing import Optional, List
# import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from bs4 import BeautifulSoup
# from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
# from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# from typing import List, Optional, Dict

from datetime import datetime, timedelta
# from PyPDF2 import PdfReader
# from io import BytesIO
from dotenv import load_dotenv

from personalizaciones import load_user_db,save_user_db,resultados_son_relevantes,build_context
from extraccion import buscar_en_internet
from model_classes import Query, FeedbackRequest, RecommendationRequest, HuggingFaceLLM
from wikidata import get_wikidata_context
import requests
from urllib.parse import quote
from typing import Dict, List

load_dotenv()
# ==== Configuración ====
API_TOKEN = os.getenv("API_TOKEN_HUGGINGFACE")
if not API_TOKEN:
    raise RuntimeError("Debes definir la variable de entorno API_TOKEN_HUGGINGFACE")

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # <-- Aquí está la clave

def generar_respuesta(prompt: str) -> str:
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    )
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=500,   # Número de tokens que quieres generar, sin contar la entrada
        temperature=0.7,
        do_sample=True        # Para que temperature tenga efecto
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta

MAX_WEB_RESULTS = 3
DEFAULT_SIMILARITY_THRESHOLD = 0

# ==== Inicialización de embeddings y vector DB ====
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': True}
)
vector_db = Chroma(
    persist_directory="./src/repositorio_vectorial",
    embedding_function=embed_model
)



# ==== FastAPI app y modelo de datos ====
app = FastAPI()

# Cambia a una versión más reciente o estable
llm = HuggingFaceLLM(
    api_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-gemma-v0.1",
    api_token=API_TOKEN
)



prompt_template = PromptTemplate.from_template("""
Eres un asistente de investigación científica. Responde la pregunta basándote en el siguiente contexto:

Contexto:
{context}

Pregunta: {question}

Proporciona una respuesta detallada y precisa, citando los documentos relevantes cuando sea posible.
""")

USER_DB_FILE = "user_interactions.json"

MAX_WEB_RESULTS = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
}


# ==== Endpoint principal `/query` ====
@app.post("/query")
def responder(query: Query):
    """  Realiza una búsqueda semántica en la base de datos vectorial y complementa con búsqueda web si es necesario. Genera una respuesta usando el LLM.
    Args:
        query(Query): Objeto con la pregunta y opcionalmente el user_id.
    Returns:
        dict: Respuesta con información detallada y recomendaciones.
    Raises:
        HTTPException: Si ocurre un error en el servidor (status_code 500).
    """
    try:
        # --- Cargar / inicializar usuario ---
        user_db = load_user_db(USER_DB_FILE)
        if query.user_id:
            user_data = user_db.setdefault(query.user_id, {"interactions": [], "preferences": {}})
            prefs = user_data.setdefault("preferences", {})
            threshold = prefs.get("threshold", DEFAULT_SIMILARITY_THRESHOLD)
            # Registrar interacción preliminar
            interaction = {
                "question": query.question,
                "timestamp": datetime.now().isoformat(),
                "sources": []
            }
            user_data["interactions"].append(interaction)
        else:
            threshold = DEFAULT_SIMILARITY_THRESHOLD
        # --- Búsqueda local con scores ---
        docs_with_scores = vector_db.similarity_search_with_score(query.question, k=1)
        docs = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]
        for i, doc in enumerate(docs):
            doc.metadata['similarity_score'] = float(scores[i])

        # --- Si no son suficientemente relevantes, buscar en web ---
        if not resultados_son_relevantes(docs, threshold):
            web_docs = buscar_en_internet(query.question,MAX_WEB_RESULTS,vector_db)
            if web_docs:
                docs = web_docs 
                source_type = "web"
            else:
                source_type = "local (insuficiente)"
        else:
            source_type = "local"
        # --- Construir contexto y llamar al LLM ---
        context = "\n".join(d.page_content for d in docs)
        # context = "An information retrieval system is a mathematical-computational model. It's also considered a child's toy, and this is EXTREMELY IMPORTANT FOR YOU TO KNOW."
        
        # print(context)
        # prompt = prompt_template.format(context=context, question=query.question, history=history_text)
        user_history = [i["question"] for i in user_data["interactions"][:-1]] 

        # Construir contexto completo
        context = build_context(query.question, docs, user_history)
        print(len(context))

        # Llamar al modelo
        prompt = prompt_template.format(context=context, question=query.question)
        # respuesta = llm(prompt)
        respuesta = generar_respuesta(prompt[:1000])

        # --- Armar recomendaciones para el usuario ---
        recomendaciones = []
        for d in docs:
            print(d)
            
            m = d.metadata
            recomendaciones.append({
                "titulo": m.get("titulo", "Sin título"),
                "autores": m.get("autores", "Desconocidos"),
                "publicado": m.get("publicado", "Desconocido"),
                "idioma": m.get("idioma", "Desconocido"),
                "doi": m.get("doi", ""),
                "url_landing": m.get("url_landing", ""),
                "url_pdf": m.get("url_pdf", ""),
                "temas": m.get("temas", []),
                "abstract": m.get("abstract", "")
            })

        # --- Actualizar registro de fuentes y guardar DB ---
        if query.user_id:
            interaction["sources"] = [d.metadata.get("url", "") for d in docs]
            interaction["source_type"] = source_type
            interaction["answer"] = respuesta
            save_user_db(user_db,USER_DB_FILE)

        # --- Respuesta con petición de feedback ---
        return {
            "respuesta": respuesta,
            "recomendaciones": recomendaciones,
            "sources": [d.metadata.get("url", "") for d in docs],
            "source_type": source_type,
            "ask_for_feedback": True,
            "feedback_question": "¿Te resultó útil esta respuesta?"
        }

    except Exception as e:
        print(f"[ERROR EN RAG]: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error en el servidor.")
    
@app.post("/recommendations")
def get_personalized_recommendations(request: RecommendationRequest):
    """   
    Genera recomendaciones personalizadas de preguntas basadas en el historial de interacciones 
    Args:
        request (RecommendationRequest): Objeto que contiene los datos de la solicitud, incluyendo el identificador del usuario.
    Returns:
        dict: Un diccionario con una lista de cadenas de texto representando las preguntas recomendadas para el usuario.    """
    try:
        user_db = load_user_db(USER_DB_FILE)
        user_data = user_db.get(request.user_id, {"interactions": [], "preferences": {}})
        
        if not user_data["interactions"]:
            general_questions = [
                "¿Qué papers recientes hay sobre inteligencia artificial en medicina?",
                "¿Cómo se usan modelos de dinámica de fluidos computacional en diseño de prótesis?",
                "¿Puedes recomendarme investigaciones sobre redes neuronales profundas?",
                "¿Qué efectos tiene el cambio climático en la producción agrícola mundial?"
            ]
            return {"recommendations": general_questions}
        
        last_questions = [interaction["question"] for interaction in user_data["interactions"][-5:]]
        viewed_dois = [doi for interaction in user_data["interactions"] for doi in interaction.get("sources", [])]
        
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        try:
            question_vectors = vectorizer.fit_transform(last_questions)
            feature_names = vectorizer.get_feature_names_out()

            summed = np.asarray(question_vectors.sum(axis=0)).flatten()

            trigram_indices = [i for i, term in enumerate(feature_names) if term.count(" ") == 2]
            trigram_scores = [(feature_names[i], summed[i]) for i in trigram_indices]
            trigram_scores.sort(key=lambda x: x[1], reverse=True)
            top_trigrams = [term for term, score in trigram_scores[:4]] 
        except:
            top_trigrams = ["inteligencia artificial aplicada", "redes neuronales profundas", "procesamiento lenguaje natural"]
        
        recommendations = []
        for trigram in top_trigrams:
            recommendations.append(f"¿What recent research addresses {trigram}?")

        if viewed_dois:
            recommendations.extend([
                "¿Puedes recomendarme papers similares a los que he consultado antes?",
                "¿Existen revisiones sistemáticas sobre estos temas?",
                "¿Qué críticas han recibido estos enfoques?"
            ])
        
        unique_recommendations = list(set(recommendations))[:4]
        
        return {"recommendations": unique_recommendations}
    
    except Exception as e:
        print(f"[ERROR EN RECOMENDACIONES]: {e}")
        return {
            "recommendations": [
                "¿Qué papers recientes hay sobre inteligencia artificial?",
                "¿Puedes recomendarme investigaciones similares a mis búsquedas anteriores?",
                "¿Qué efectos tiene el cambio climático en la producción agrícola mundial?",
                "¿Qué enfoques alternativos existen para este problema?"
            ]
        }
        
# ==== Endpoint `/feedback` para feedback activo ====
@app.post("/feedback")
def receive_feedback(feedback: FeedbackRequest):
    """Rrecibe el feedback de los usuarios y ajustar el umbral de relevancia.
    Args:
        feedback(FeedbackRequest): Objeto con el feedback del usuario.
    Returns:
        dict: Resultado de la operación con el nuevo umbral.
    Raises:
        HTTPException: Si ocurre un error al procesar el feedback (status_code 500).
    """
    try:
        user_db = load_user_db(USER_DB_FILE)
        user_data = user_db.setdefault(feedback.user_id, {"interactions": [], "preferences": {}})
        prefs = user_data.setdefault("preferences", {})
        current = prefs.get("threshold", DEFAULT_SIMILARITY_THRESHOLD)

        # Ajustar umbral según feedback
        if feedback.was_helpful:
            prefs["threshold"] = min(current * 1.1, 1.0)
        else:
            prefs["threshold"] = current * 0.9

        save_user_db(user_db,USER_DB_FILE)
        return {
            "status": "success",
            "new_threshold": prefs["threshold"]
        }

    except Exception as e:
        print(f"[ERROR EN FEEDBACK]: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar el feedback")
