import os
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from io import BytesIO
import pandas as pd
import time
from dotenv import load_dotenv
from typing import List, Dict, Optional



def fetch_crossref_papers(existing_dois,query: str, max_results: int = 100, years_back: int = 5, offset: int = 0):
    """
    Busca artículos científicos en Crossref con DOI y filtro por fecha.
    """
    url = "https://api.crossref.org/works" 
    fecha_limite = (datetime.now() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")

    params = {
        'query': query,
        'rows': max_results,
        'offset': offset,
        'filter': f"from-pub-date:{fecha_limite}"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("message", {}).get("items", []):
            if item.get("language", "en") != "en":
                continue

            title = item.get("title", [""])[0]
            doi = item.get("DOI")
            if not doi or doi in existing_dois:
                continue

            landing_page = f"https://doi.org/{doi}" 
            authors = item.get("author", [])
            authors_str = ", ".join([f"{a.get('given', '')} {a.get('family', '')}" for a in authors])

            published_parts = item.get("published-print", item.get("published-online", {})).get("date-parts", [])
            published = "-".join(map(str, published_parts[0])) if published_parts else ""

            abstract = item.get("abstract", "")
            if abstract:
                abstract = BeautifulSoup(abstract, "html.parser").get_text()

            papers.append({
                "titulo": title,
                "autores": authors_str,
                "publicado": published,
                "idioma": item.get("language", "en"),
                "doi": doi,
                "url": landing_page,
                "abstract": abstract,
            })
        return papers
    except Exception as e:
        print(f"❌ Error en Crossref: {e}")
        return []

def get_open_access_pdf(doi: str, email):
    """Busca un PDF abierto usando Unpaywall."""
    url = f"https://api.unpaywall.org/v2/{doi}" 
    try:
        response = requests.get(url, params={"email": email}).json()
        if response.get("is_oa"):
            return response.get("best_oa_location", {}).get("url_for_pdf")
    except Exception as e:
        print(f"❌ Error en Unpaywall: {e}")
    return None

def extract_text_from_pdf(pdf_url: str):
    """Extrae texto de un PDF dado su URL."""
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        if "application/pdf" not in response.headers.get("Content-Type", ""):
            return None
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        print(f"❌ Error leyendo PDF: {e}")
        return None

def find_direct_pdf_link(url: str):
    """Busca enlaces a PDFs dentro de la página."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            if ".pdf" in link["href"].lower():
                return requests.compat.urljoin(url, link["href"])
    except Exception as e:
        print(f"❌ Error buscando PDF manualmente: {e}")
    return None

