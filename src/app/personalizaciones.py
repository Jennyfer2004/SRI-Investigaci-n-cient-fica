import os
import json
from datetime import datetime, timedelta
from wikidata import get_wikidata_context


def load_user_db(USER_DB_FILE):
    """Carga la base de datos de interacciones de usuario desde un archivo JSON.
    Returns:
        dict: Diccionario con los datos de usuarios.
    """
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_db(db,USER_DB_FILE):
    """Guarda la base de datos de interacciones de usuario en un archivo JSON.
    Args:
        db(dict): Diccionario con los datos de usuarios a guardar.
    """
    with open(USER_DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)

def resultados_son_relevantes(docs, threshold):
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
        if year_str.isdigit() and (current_year - int(year_str) <= 20):
            return True

    return False

def build_context(query: str, docs, user_history) -> str:
    """Construye el contexto para el LLM combinando múltiples fuentes"""
    
    # 1. Información de documentos relevantes
    document_context = "\n".join(
        f"Documento {i+1}:\nContenido: {doc.page_content}\nMetadatos: {doc.metadata}"
        for i, doc in enumerate(docs))
    
    # 2. Historial de consultas del usuario (últimas 3)
    history_context = "Historial de consultas recientes:\n" + "\n".join(
        f"- {q}" for q in user_history[-3:]) if user_history else ""
    
    # 3. Información de Wikidata
    wikidata_context = get_wikidata_context(query)
    
    # 4. Contexto del sistema (puedes personalizar esto)
    system_context = (
        "Eres un asistente de investigación científica. "
        "Debes responder preguntas basándote en los documentos proporcionados, "
        "considerando el historial de búsqueda del usuario."
    )
    
    return (
        f"{system_context}\n\n"
        f"{history_context}\n\n"
        f"Información relevante de bases de datos:\n{document_context}\n\n"
        f"Contexto de conocimiento general:\n{wikidata_context}"
    )