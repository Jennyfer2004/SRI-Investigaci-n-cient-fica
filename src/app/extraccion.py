from datetime import datetime, timedelta
import time

import os
from io import BytesIO
from PyPDF2 import PdfReader

import requests
from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd
import re

def normalize_text(text):
    """ 
    Normaliza un texto eliminando caracteres especiales y estandarizando formatos.
    Args:
        text(str): Texto sin procesar
    Return:
        str: Texto preprocesado para caracteres especiales.
    """
    if type(text)==float:
        print(text)
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r'\s+', ' ', text)
    return text
urls=[]
count=0
def remove_sections(text):
    """    
    Elimina secciones específicas del contenido del artículo como Referencias y Agradecimientos, así como todo lo que contenga después 
    Args:
        text(str): Contenido del artículo para eliminar secciones
    Return:    
        str: El texto anterior eliminando las secciones anteriores
    """
    global count
    count+=1
    after_refs = re.split(r'\n(references|bibliography|acknowledg(e)?ments)\b', text, flags=re.IGNORECASE)  

    return after_refs[0]


def clean_text(text):
    """
    Realiza limpieza del texto eliminando caracteres invisibles y espacios múltiples.
    Args:
        text (str): Texto a limpiar.
    Return:    
        str: Texto limpio con espaciado uniforme

    """
   
    if not isinstance(text, str): 
        return ""
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
}

def fetch_crossref_papers(query: str, max_results=5, years_back=5):
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

def get_open_access_pdf(doi: str, email: str = os.getenv("email")):
    """Obtiene el link público del PDF desde Unpaywall."""
    try:
        response = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={email}", timeout=10).json()
        return response.get("best_oa_location", {}).get("url_for_pdf")
    except Exception as e:
        print(f"❌ Error en Unpaywall: {e}")
        return None

def extract_text_from_pdf(pdf_url: str):
    """Extrae texto de un PDF remoto."""
    try:
        response = requests.get(pdf_url, timeout=10, headers=HEADERS)
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        print(f"❌ Error extrayendo PDF: {e}")
        return None

def extract_content_from_html(url: str):
    """Extrae contenido de una página web."""
    try:
        response = requests.get(url, timeout=20, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        return " ".join(p.get_text().strip() for p in soup.find_all("p"))[:5000]
    except Exception as e:
        print(f"❌ Error extrayendo HTML: {e}")
        return None

def buscar_en_internet(query: str, num_results: int ):
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



def buscar_en_internet(query: str, num_results: int,vector_db):
    """
    Busca artículos científicos y devuelve Documentos listos para RAG con el mismo formato que ChromaDB.
    """
    # Usar el mismo text splitter que en tu ChromaDB
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", r"(?<=\. )", " "],
        length_function=len,
        is_separator_regex=True
    )

    documents = []
    crossref_results = fetch_crossref_papers(query, max_results=num_results, years_back=5)

    for paper in crossref_results:
        pdf_url = get_open_access_pdf(paper["doi"])
        raw_content = extract_text_from_pdf(pdf_url) if pdf_url else extract_content_from_html(paper["url"])
        
        # Aplicar el mismo preprocesamiento que en tu ChromaDB
        clean_title = clean_text(paper["titulo"])
        clean_abstract = clean_text(paper["abstract"])
        clean_content = remove_sections(raw_content)
        clean_content = normalize_text(clean_content)
        clean_content = clean_text(clean_content)
        
        # Filtrar contenido muy corto (como haces en ChromaDB)
        if len(clean_content) <= 200:
            continue
            
        # Crear el documento con la misma estructura
        full_content = f"{clean_abstract}\n\n{clean_content}"
        
        # Metadatos consistentes con ChromaDB
        metadata = {
            'titulo': clean_title,
            'doi': paper["doi"],
            'autores': paper["autores"] or "Desconocidos",
            'publicado': paper["publicado"] or "Desconocido",
            'idioma': paper["idioma"] or "Desconocido",
            'url_pdf': pdf_url or "",
            'url_landing': paper["url"],
            'abstract': clean_abstract,
            'fuente': 'web',
            'consultado_en': datetime.now().isoformat()
        }
        
        base_doc = Document(page_content=full_content, metadata=metadata)
        splits = text_splitter.split_documents([base_doc])
        
        # Añadir fragmento_id como en ChromaDB
        for j, split in enumerate(splits):
            split.metadata['fragmento_id'] = f"{metadata['doi']}_part{j+1}"
            # Añadir score temporal como 1.0 (máxima similitud) ya que son nuevos
            split.metadata['similarity_score'] = 1.0
        
        documents.extend(splits)
        time.sleep(1)  # Politeness delay

    # Si encontramos documentos, los añadimos a ChromaDB
    if documents:
        # Añadir los textos y metadatos a ChromaDB
        vector_db.add_texts(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
        
        # Devolver los documentos con sus metadatos (incluyendo el similarity_score temporal)
        return documents[-1]
    
    return []

