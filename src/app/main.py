import os
import json
import requests
from typing import Optional, List
import time

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
from typing import List, Optional, Dict

from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
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
    persist_directory="./src/repositorio_vectorial",
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
    
class RecommendationRequest(BaseModel):
    user_id: str

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
        print("sin docs returnados")
        return False
    if docs[0].metadata.get('similarity_score', 0) < threshold:
        return False
    current_year = datetime.now().year
    for doc in docs:
        year_str = doc.metadata.get('publicado', '')
        if year_str.isdigit() and (current_year - int(year_str) <= 5):
            return True

    return False


MAX_WEB_RESULTS = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
}

def fetch_crossref_papers(query: str, max_results=5, years_back=5) -> List[Dict]:
    """Busca artículos en Crossref con filtro por año."""
    url = "https://api.crossref.org/works" 
    fecha_limite = (datetime.now() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")

    params = {
        'query': query,
        'rows': max_results,
        'filter': f"from-pub-date:{fecha_limite}"
    }

    try:
        response = requests.get(url, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("message", {}).get("items", []):
            title = item.get("title", [""])[0]
            doi = item.get("DOI")
            landing_page = f"https://doi.org/{doi}" 

            authors = ", ".join([f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])])
            published = "-".join(map(str, item.get("published-print", {}).get("date-parts", [[]])[0])) or ""

            abstract = BeautifulSoup(item.get("abstract", ""), "html.parser").get_text() if item.get("abstract") else ""
            language = item.get("language", "en")
            if not abstract or not doi:
                continue
            papers.append({
                "titulo": title,
                "autores": authors,
                "publicado": published,
                "idioma": language,
                "doi": doi,
                "url": landing_page,
                "abstract": abstract,
            })
        return papers
    except Exception as e:
        print(f"❌ Error en Crossref: {e}")
        return []

def get_open_access_pdf(doi: str, email: str = os.getenv("email")) -> Optional[str]:
    """Obtiene el link público del PDF desde Unpaywall."""
    try:
        response = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={email}", timeout=10).json()
        return response.get("best_oa_location", {}).get("url_for_pdf")
    except Exception as e:
        print(f"❌ Error en Unpaywall: {e}")
        return None

def extract_text_from_pdf(pdf_url: str) -> Optional[str]:
    """Extrae texto de un PDF remoto."""
    try:
        response = requests.get(pdf_url, timeout=10, headers=HEADERS)
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        print(f"❌ Error extrayendo PDF: {e}")
        return None

def extract_content_from_html(url: str) -> Optional[str]:
    """Extrae contenido de una página web."""
    try:
        response = requests.get(url, timeout=10, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        return " ".join(p.get_text().strip() for p in soup.find_all("p"))[:5000]
    except Exception as e:
        print(f"❌ Error extrayendo HTML: {e}")
        return None

def buscar_en_internet(query: str, num_results: int = MAX_WEB_RESULTS) -> List[Document]:
    """
    Busca artículos científicos y devuelve Documentos listos para RAG.
    """
    documents = []

    crossref_results = fetch_crossref_papers(query, max_results=num_results, years_back=5)

    for paper in crossref_results:
        pdf_url = get_open_access_pdf(paper["doi"])
        content = extract_text_from_pdf(pdf_url) if pdf_url else extract_content_from_html(paper["url"])


        doc = Document(
            page_content=content,
            metadata={
                "titulo": paper["titulo"],
                "autores": paper["autores"] or "Desconocidos",
                "publicado": paper["publicado"] or "Desconocido",
                "idioma": paper["idioma"] or "Desconocido",
                "doi": paper["doi"],
                "url": paper["url"],
                "abstract": paper["abstract"],
                "fuente": "ciencia",
                "consultado_en": datetime.now().isoformat()
            }
        )
        documents.append(doc)
        time.sleep(1)

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
    
@app.post("/recommendations")
def get_personalized_recommendations(request: RecommendationRequest):
    try:
        user_db = load_user_db()
        user_data = user_db.get(request.user_id, {"interactions": [], "preferences": {}})
        
        # Si no hay historial, devolver recomendaciones generales
        if not user_data["interactions"]:
            general_questions = [
                "¿Qué papers recientes hay sobre inteligencia artificial en medicina?",
                "¿Cuáles son los últimos avances en procesamiento de lenguaje natural?",
                "¿Puedes recomendarme investigaciones sobre redes neuronales profundas?",
                "¿Qué papers importantes existen sobre ética en IA?"
            ]
            return {"recommendations": general_questions}
        
        # Extraer temas de interés del usuario
        last_questions = [interaction["question"] for interaction in user_data["interactions"][-5:]]
        viewed_dois = [doi for interaction in user_data["interactions"] for doi in interaction.get("sources", [])]
        
        # Vectorizar preguntas para encontrar temas comunes
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        try:
            question_vectors = vectorizer.fit_transform(last_questions)
            feature_names = vectorizer.get_feature_names_out()

            # Sumar los pesos de TF-IDF para cada término
            summed = np.asarray(question_vectors.sum(axis=0)).flatten()

            # Filtrar solo trigramas (palabras separadas por dos espacios)
            trigram_indices = [i for i, term in enumerate(feature_names) if term.count(" ") == 2]

            # Extraer trigramas y sus puntuaciones
            trigram_scores = [(feature_names[i], summed[i]) for i in trigram_indices]

            # Ordenar por score descendente
            trigram_scores.sort(key=lambda x: x[1], reverse=True)

            # Tomar el trigram más representativo (mayor score)
            top_trigrams = [term for term, score in trigram_scores[:4]]  # puedes cambiar a [:3] si quieres más
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
        
        # Eliminar duplicados y limitar a 4 recomendaciones
        unique_recommendations = list(set(recommendations))[:4]
        
        return {"recommendations": unique_recommendations}
    
    except Exception as e:
        print(f"[ERROR EN RECOMENDACIONES]: {e}")
        return {
            "recommendations": [
                "¿Qué papers recientes hay sobre inteligencia artificial?",
                "¿Puedes recomendarme investigaciones similares a mis búsquedas anteriores?",
                "¿Cuáles son los papers más citados en este área?",
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
