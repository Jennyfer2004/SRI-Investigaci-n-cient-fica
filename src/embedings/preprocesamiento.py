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