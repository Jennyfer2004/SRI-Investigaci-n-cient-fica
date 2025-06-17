from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from typing import Optional, List, Any, Dict
from langchain.llms.base import LLM
from pydantic import Field
import requests
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  
import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.docstore.document import Document
api_token=os.getenv("api_token_huggingface")
MIN_SIMILARITY_THRESHOLD = 0.25
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={'normalize_embeddings': True}
)

vector_db = Chroma(
    persist_directory="./repositorio_vectorial",
    embedding_function=embed_model
)


class HuggingFaceLLM(LLM):
    api_url: str = Field(...)
    api_token: str = Field(...)
    max_new_tokens: int = Field(default=500)
    temperature: float = Field(default=0.7)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature
            }
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        else:
            return str(data)

    @property
    def _llm_type(self) -> str:
        return "huggingface_custom"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "api_url": self.api_url,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }

app = FastAPI()

llm = HuggingFaceLLM(
    api_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    api_token=api_token
)

# Plantilla mejorada para el prompt
prompt_template = PromptTemplate.from_template("""
Eres un asistente de investigación científica. Responde la pregunta basándote en el siguiente contexto:

Contexto:
{context}

Pregunta: {question}

Proporciona una respuesta detallada y precisa, citando los documentos relevantes cuando sea posible.
""")

USER_DB_FILE = "user_interactions.json"

# Cargar o inicializar la base de datos de usuarios
def load_user_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_db(user_db):
    with open(USER_DB_FILE, 'w') as f:
        json.dump(user_db, f)

# Función para evaluar relevancia de resultados
def resultados_son_relevantes(docs: List[Document], query: str) -> bool:
    """Evalúa si los resultados son relevantes y actualizados"""
    if not docs:
        return False
    
    # Verificar similitud del mejor resultado
    if docs[0].metadata.get('similarity_score', 0) < MIN_SIMILARITY_THRESHOLD:
        return False
    
    # Verificar antigüedad (más de 2 años)
    current_year = datetime.now().year
    for doc in docs:
        year_str = doc.metadata.get('publicado', '')
        if year_str and year_str.isdigit():
            if current_year - int(year_str) <= 2:
                return True
    return False

# Modelos de datos
class Query(BaseModel):
    question: str
    user_id: Optional[str] = None  # Nuevo campo para identificar usuarios

class RecommendationRequest(BaseModel):
    user_id: str

class FeedbackRequest(BaseModel):
    user_id: str
    question: str
    was_helpful: bool
    feedback_text: Optional[str] = None

# Endpoint principal para consultas
@app.post("/query")
def responder(query: Query):
    try:
        # Registrar la interacción del usuario
        if query.user_id:
            user_db = load_user_db()
            user_interactions = user_db.get(query.user_id, {"interactions": [], "preferences": {}})
            
            interaction_record = {
                "question": query.question,
                "timestamp": datetime.now().isoformat(),
                "sources": []
            }
            
            user_interactions["interactions"].append(interaction_record)
            user_db[query.user_id] = user_interactions
            save_user_db(user_db)

        # Búsqueda de documentos relevantes
        docs = vector_db.similarity_search(query.question, k=5)
        context = "\n".join(doc.page_content for doc in docs)
        
        # Generar respuesta
        prompt = prompt_template.format(context=context, question=query.question)
        respuesta = llm(prompt)
        
        # Preparar recomendaciones de documentos
        recomendaciones = [
            {
                "titulo": doc.metadata.get("titulo", "Sin título"),
                "autores": doc.metadata.get("autores", "Desconocidos"),
                "publicado": doc.metadata.get("publicado", "Desconocido"),
                "idioma": doc.metadata.get("idioma", "Desconocido"),
                "doi": doc.metadata.get("doi", ""),
                "url_landing": doc.metadata.get("url_landing", ""),
                "url_pdf": doc.metadata.get("url_pdf", ""),
                "temas": doc.metadata.get("temas", []),
                "abstract": doc.metadata.get("abstract", "")
            } for doc in docs
        ]
        
        # Registrar los documentos consultados
        if query.user_id:
            interaction_record["sources"] = [doc.metadata.get("doi", "") for doc in docs]
            save_user_db(user_db)

        return {
            "respuesta": respuesta,
            "recomendaciones": recomendaciones,
            "sources": [doc.metadata.get("doi", "") for doc in docs]
        }

    except Exception as e:
        print(f"[ERROR EN RAG]: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error en el servidor.")

# Nuevo endpoint para recomendaciones personalizadas
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

        
        # # Generar preguntas recomendadas basadas en los intereses
        # recommendations = []
        # for term in top_trigrams:
        #     if term in ["ai", "inteligencia", "artificial"]:
        #         recommendations.append(f"¿Cuáles son los últimos avances en inteligencia artificial sobre {top_trigrams[1]}?")
        #     elif term in ["aprendizaje", "learning"]:
        #         recommendations.append(f"¿Qué técnicas de {term} automático se aplican actualmente en investigación?")
        #     else:
        #         recommendations.append(f"¿Qué papers recientes existen sobre {term}?")
        
        # Añadir preguntas de seguimiento basadas en documentos vistos
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

# Endpoint para recibir feedback
@app.post("/feedback")
def receive_feedback(feedback: FeedbackRequest):
    try:
        user_db = load_user_db()
        user_data = user_db.get(feedback.user_id, {"interactions": [], "preferences": {}})
        
        # Actualizar preferencias basadas en el feedback
        if "preferences" not in user_data:
            user_data["preferences"] = {}
        
        # Aquí podrías añadir lógica más sofisticada para analizar el feedback
        if feedback.was_helpful:
            user_data["preferences"]["last_helpful_question"] = feedback.question
        
        user_db[feedback.user_id] = user_data
        save_user_db(user_db)
        
        return {"status": "success", "message": "Feedback recibido correctamente"}
    
    except Exception as e:
        print(f"[ERROR EN FEEDBACK]: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar el feedback")

# Función para analizar temas de interés (puede ser mejorada)
def analyze_user_interests(questions: List[str]) -> List[str]:
    # Implementación básica - en producción usarías NLP más avanzado
    vectorizer = TfidfVectorizer(stop_words=[ 'english'],ngram_range=(1,3))
    try:
        X = vectorizer.fit_transform(questions)
        features = vectorizer.get_feature_names_out()
        sums = X.sum(axis=0)
        top_indices = np.argsort(sums)[0, -3:].tolist()[0]
        return [features[i] for i in top_indices]
    except:
        return ["investigación", "ciencia", "tecnología"]