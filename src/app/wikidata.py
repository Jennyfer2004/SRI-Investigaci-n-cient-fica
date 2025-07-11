# Añade estas importaciones
import requests
from urllib.parse import quote
from typing import Dict, List
import spacy

# Configuración de Wikidata
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

nlp = spacy.load("en_core_web_sm")  # English model for term extraction

def search_wikidata_entity(term: str):
    """Buscar un término en Wikidata (solo en inglés)"""
    params = {
        "action": "wbsearchentities",
        "search": term,
        "language": "en",
        "format": "json"
    }
    try:
        response = requests.get(WIKIDATA_API, params=params, timeout=5).json()
        if response.get("search"):
            return {
                "id": response["search"][0]["id"],
                "label": response["search"][0]["label"],
                "description": response["search"][0].get("description", "No description available"),
                "url": f"https://www.wikidata.org/wiki/{response['search'][0]['id']}"
            }
    except Exception as e:
        print(f"Error querying Wikidata: {e}")
    return None

def get_wikidata_properties(qid: str):
    """Obtener propiedades de entidad de Wikidata"""
    query = f"""
    SELECT ?prop ?propLabel ?value ?valueLabel WHERE {{
      wd:{qid} ?prop ?value.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
      FILTER(?prop IN (wdt:P31, wdt:P61, wdt:P366, wdt:P279))
    }}
    """
    try:
        headers = {"Accept": "application/json"}
        response = requests.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=10
        )
        return response.json()["results"]["bindings"]
    except Exception as e:
        print(f"Error in SPARQL query: {e}")
        return []

def extract_scientific_terms(text: str) -> list:
    """Extraer términos científicos mediante PNL """
    doc = nlp(text)
    terms = []
    # Using noun chunks and entities for better English term extraction
    terms.extend([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1])
    scientific_entity_labels = ["ORG", "PRODUCT", "GPE", "EVENT", "WORK_OF_ART"]
    terms.extend([ent.text for ent in doc.ents if ent.label_ in scientific_entity_labels])
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 5:
            terms.append(token.text)
    return list(set(terms))

def get_wikidata_context(query: str) -> str:
    """Obtener el contexto de Wikidata relevante para una consulta"""
    terms = extract_scientific_terms(query)
    wikidata_context = []
    
    for term in terms:
        entity = search_wikidata_entity(term)
        if entity:
            properties = get_wikidata_properties(entity["id"])
            props_text = "\n".join(
                f"{p['propLabel']['value']}: {p['valueLabel']['value']}" 
                for p in properties[:3]
            ) if properties else "No additional properties found"
            
            wikidata_context.append(
                f"Concept: {entity['label']}\n"
                f"Description: {entity['description']}\n"
            )
    
    return "\n\n".join(wikidata_context) if wikidata_context else ""
