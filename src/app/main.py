import os
import json
import requests
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# ==== Configuración ====
API_TOKEN = os.getenv("API_TOKEN_HUGGINGFACE")
if not API_TOKEN:
    raise RuntimeError("Debes definir la variable de entorno API_TOKEN_HUGGINGFACE")

MAX_WEB_RESULTS = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.25

# ==== Inicialización de embeddings y vector DB ====
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': True}
)
vector_db = Chroma(
    persist_directory="./repositorio_vectorial",
    embedding_function=embed_model
)

# ==== Definición del LLM personalizado ====
class HuggingFaceLLM(LLM):
    """Implementación personalizada de LLM que utiliza la API de Hugging Face.
    
    Args:
        api_url(str): URL del endpoint de la API de Hugging Face.
        api_token(str): Token de autenticación para la API.
        max_new_tokens(int): Número máximo de tokens a generar (default: 500).
        temperature(float): Parámetro de temperatura para la generación (default: 0.7).
    """
    api_url: str = Field(...)
    api_token: str = Field(...)
    max_new_tokens: int = Field(default=500)
    temperature: float = Field(default=0.7)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Genera texto a partir de un prompt usando la API de Hugging Face.
        
        Args:
            prompt (str): Texto de entrada para la generación.
            stop(Optional[List[str]]): Lista de secuencias para detener la generación.    
        Returns:
            str: Texto generado por el modelo.
        """
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature
            }
        }
        r = requests.post(self.api_url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return str(data)

    @property
    def _llm_type(self) -> str:
        return "huggingface_custom"

    @property
    def _identifying_params(self) -> dict:
        """Parámetros identificadores del modelo."""
        return {
            "api_url": self.api_url,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }

# ==== FastAPI app y modelo de datos ====
app = FastAPI()

llm = HuggingFaceLLM(
    api_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
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

class Query(BaseModel):
    """Modelo de datos para las consultas de los usuarios
    Attributes:
        question(str): Pregunta del usuario.
        user_id(Optional[str]): Identificador del usuario (opcional).
    """
    question: str
    user_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    """Modelo de datos para el feedback de los usuarios.
    Attributes:
        user_id(str): Identificador del usuario.
        question(str): Pregunta original.
        was_helpful (bool): Indica si la respuesta fue útil.
        feedback_text (Optional[str]): Comentario adicional del usuario (opcional).
    """
    user_id: str
    question: str
    was_helpful: bool
    feedback_text: Optional[str] = None

# ==== Funciones de utilidad ====
def load_user_db():
    """Carga la base de datos de interacciones de usuario desde un archivo JSON.
    Returns:
        dict: Diccionario con los datos de usuarios.
    """
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_db(db: dict):
    """Guarda la base de datos de interacciones de usuario en un archivo JSON.
    Args:
        db(dict): Diccionario con los datos de usuarios a guardar.
    """
    with open(USER_DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)

def resultados_son_relevantes(
    docs: List[Document], threshold: float
) -> bool:
    """Evalúa si los resultados son relevantes y actualizados.
    Args:
        docs (List[Document]): Lista de documentos a evaluar.
        threshold (float): Umbral mínimo de similitud para considerar relevante.
    Returns:
        bool: True si hay documentos relevantes y actualizados, False en caso contrario.
        """
    if not docs:
        return False
    if docs[0].metadata.get('similarity_score', 0) < threshold:
        return False
    current_year = datetime.now().year
    for doc in docs:
        year_str = doc.metadata.get('publicado', '')
        if year_str.isdigit() and (current_year - int(year_str) <= 5):
            return True
    return False

def buscar_en_internet(query: str, num_results: int = MAX_WEB_RESULTS) -> List[Document]:
    """Busca información en internet usando DuckDuckGo y extrae contenido de las páginas.
    Args:
        query(str): Término de búsqueda.
        num_results(int): Número máximo de resultados a retornar (default: MAX_WEB_RESULTS).
    Returns:
        List[Document]: Lista de documentos con el contenido extraído de las páginas web.
    """
    documents: List[Document] = []
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=num_results))
            for r in resultados:
                try:
                    resp = requests.get(r['href'], timeout=10)
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    content = ' '.join(p.get_text().strip() for p in paragraphs)[:5000]
                    doc = Document(
                        page_content=content,
                        metadata={
                            'titulo': r.get('title', ''),
                            'url': r['href'],
                            'fuente': 'web',
                            'publicado': datetime.now().strftime("%Y"),
                            'consultado_en': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
                except Exception:
                    continue
    except Exception:
        pass
    return documents

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
        user_db = load_user_db()
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
        docs_with_scores = vector_db.similarity_search_with_score(query.question, k=5)
        docs = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]
        for i, doc in enumerate(docs):
            doc.metadata['similarity_score'] = float(scores[i])

        # --- Si no son suficientemente relevantes, buscar en web ---
        if not resultados_son_relevantes(docs, threshold):
            web_docs = buscar_en_internet(query.question)
            if web_docs:
                vector_db.add_texts([d.page_content for d in web_docs],
                                    [d.metadata for d in web_docs])
                docs = web_docs
                source_type = "web"
            else:
                source_type = "local (insuficiente)"
        else:
            source_type = "local"

        # --- Construir contexto y llamar al LLM ---
        context = "\n".join(d.page_content for d in docs)
        prompt = prompt_template.format(context=context, question=query.question)
        respuesta = llm(prompt)

        # --- Armar recomendaciones para el usuario ---
        recomendaciones = []
        for d in docs:
            m = d.metadata
            recomendaciones.append({
                "titulo": m.get("titulo", "Sin título"),
                "autores": m.get("autores", "Desconocidos"),
                "publicado": m.get("publicado", "Desconocido"),
                "idioma": m.get("idioma", "Desconocido"),
                "doi": m.get("doi", ""),
                "url": m.get("url", ""),
                "abstract": m.get("abstract", "")
            })

        # --- Actualizar registro de fuentes y guardar DB ---
        if query.user_id:
            interaction["sources"] = [d.metadata.get("url", "") for d in docs]
            interaction["source_type"] = source_type
            save_user_db(user_db)

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
        user_db = load_user_db()
        user_data = user_db.setdefault(feedback.user_id, {"interactions": [], "preferences": {}})
        prefs = user_data.setdefault("preferences", {})
        current = prefs.get("threshold", DEFAULT_SIMILARITY_THRESHOLD)

        # Ajustar umbral según feedback
        if feedback.was_helpful:
            prefs["threshold"] = min(current * 1.1, 1.0)
        else:
            prefs["threshold"] = current * 0.9

        save_user_db(user_db)
        return {
            "status": "success",
            "new_threshold": prefs["threshold"]
        }

    except Exception as e:
        print(f"[ERROR EN FEEDBACK]: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar el feedback")
