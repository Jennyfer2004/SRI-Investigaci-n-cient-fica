import re
import requests
import time

from bs4 import BeautifulSoup
from io import BytesIO
from PyPDF2 import PdfReader

def remove_sections(text,urls):
    """
    Remover sección de referencias y añadir las url a la lista 
    Args:
        text(str): texto del artićulo
        urls(list): lista global de referencias encontradas
    Returns:
        str: sin referencias
    """
    global a
    a+=1
    after_refs = re.split(r'\n(references|bibliography|acknowledg(e)?ments)\b', text, flags=re.IGNORECASE)  

    url_patron= r'https?://\S+|www\.\S+'

    for i in after_refs:
        url=[]
        if not i:
            continue
        if len(i)==0:
            continue
        url = re.findall(url_patron, i)
        url1 = [re.sub(r'[).,;]+$', '', url_sin_parentesis.strip()) for url_sin_parentesis in url if url_sin_parentesis.strip()]
        if url1:
            for k in url1:
                urls.append(k)
    print(f"Total de papers visitados {a}")
    return after_refs[0]

def obtener_datos_unpaywall(doi, email):
    """
    Obtiene información sobre la disponibilidad de doi usando la api de unpaywall.
    Args:
        doi(str): el identificador de un artículo .
        email(str): correo electrónico registrado para usar la API de Unpaywall.
    Returns:
        dict or None: json de la API si es exitosa, None en caso de error.
   
   """
    url = f"https://api.unpaywall.org/v2/{doi}" 
    params = {
        "email": email 
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Error de conexión para DOI {doi}: {str(e)}")
        time.sleep(5) 
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error HTTP para DOI {doi}: {str(e)}")
        return None
def extract_text_from_pdf(pdf_url):
    """
    Extrae el texto de un archivo PDF desde una url
    Args:
        pdf_url(str): url del pdf
    Returns:
        str: Texto extraído del PDF
    """
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        time.sleep(1)

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            return None
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            return "\n".join(filter(None, (p.extract_text() for p in reader.pages))).strip()
    except Exception as e:
        print(f"❌ Error leyendo PDF: {e}")
        return None
    
def extraer_info_relevante(datos):
    """
    Extrae información sobre tos artículos
    Args:
        datos (dict): datos obtenidos desde la api de unpaywall.
    Returns:
        dict : Diccionario con los datos relevantes
    """
    title = datos.get("title", "")
    doi = datos.get("doi", "")
    if not datos:
        return None
    if not datos.get("best_oa_location"):
        return None 
    landing_page = datos.get("best_oa_location", {}).get("url_for_landing_page", "")
    
    authors = datos.get("z_authors", [])
    authors_str = ", ".join([a.get("raw_author_name", "") for a in authors])
    
    published = datos.get("published_date", "")
    language = datos.get("language", "en")
    
    abstract = datos.get("abstract", "No abstract available.")

    return {
        "title": title,
        "doi": doi,
        "landing_page": landing_page,
        "authors": authors_str,
        "published": published,
        "abstract": abstract,
        "language": language
    }
def find_direct_pdf_link(url):
    """
    Intenta encontrar un enlace directo a PDF en una página web
    Args:
        url(str): url de la página donde buscar el enlace del pdf
    Returns:
        str: url completa del PDF
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                if ".pdf" in link["href"].lower():
                    return requests.compat.urljoin(url, link["href"])
    except Exception as e:
        print(f"❌ PDF manual falló: {e}")
    return None
def get_open_access_pdf(doi, email):
    """
    Busca con la api unpaywall el contenidos de los artículos
    Args:
       doi(str): identificador del artículo
       email(str): correo electrónico registrado para usar la API de Unpaywall.
    Returns:
       str: url del pdf
    """
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": email}
    try:
        response = requests.get(url, params=params).json()
        if response.get("is_oa"):
            return response.get("best_oa_location", {}).get("url_for_pdf")
    except Exception as e:
        print(f"❌ Error en Unpaywall: {e}")
    return None

def extract_dois_from_text(text):
    """
    Extrae los dois encontrados dentro de un texto dado.
    Args:
        text (str): Texto donde se realizará la búsqueda de dois.
    Returns:
        list: Lista de dois encontrados en el texto.
    """
    # DOI estándar: 10.<registro>/<sufijo>
    doi_pattern = r"10.\d{4,9}/[-._;()/:A-Z0-9]+"
    return re.findall(doi_pattern, text, flags=re.IGNORECASE)

def is_doi(url):
    """
    Verifica si la url es un doi o lo contiene.
    Args:
        url (str): url que se va a comprobar.
    Returns:
        bool: True si la url contiene un doi, False en caso contrario.
        """
    return "doi.org/10" in url.lower()