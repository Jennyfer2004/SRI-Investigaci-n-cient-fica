from query import QUERIES
from utils import fetch_crossref_papers, get_open_access_pdf, find_direct_pdf_link, extract_text_from_pdf

import time
import pandas as pd 
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

# Configuración desde .env
EMAIL = os.getenv("email")
MAX_PER_QUERY = 250
TOTAL_MAX = 2000
YEARS_BACK = 5

# Archivos de control
VISITED_DOIS_FILE = "dois_visitados.txt"
OUTPUT_FILE = "papers_guardados.csv"

# Cargar DOIs ya procesados
existing_dois = set()
if os.path.exists(VISITED_DOIS_FILE):
    with open(VISITED_DOIS_FILE, "r") as f:
        existing_dois.update(line.strip() for line in f)

results = []
if os.path.exists(OUTPUT_FILE):
    df_existente = pd.read_csv(OUTPUT_FILE)
    results = df_existente.to_dict(orient="records")

total_collected = len(results)


for query in QUERIES:
    print(f"\n🔍 Consultando: {query}")
    for offset in range(0, MAX_PER_QUERY, 100):
        if total_collected >= TOTAL_MAX:
            break

        papers = fetch_crossref_papers(existing_dois,query=query, max_results=100, years_back=YEARS_BACK, offset=offset)

        with open(VISITED_DOIS_FILE, "a") as f_dois:
            for paper in papers:
                if total_collected >= TOTAL_MAX:
                    break

                doi = paper["doi"]
                if doi in existing_dois:
                    continue

                existing_dois.add(doi)
                landing_url = paper["url"]
                pdf_url = get_open_access_pdf(doi)

                if not pdf_url:
                    pdf_url = find_direct_pdf_link(landing_url)

                content = extract_text_from_pdf(pdf_url) if pdf_url else None
                paper["contenido"] = content

                if content:
                    results.append(paper)
                    f_dois.write(doi + "\n")
                    total_collected += 1
                    print(f"✅ ({total_collected}) {paper['titulo'][:60]}...")
                    pd.DataFrame([paper]).to_csv(
                        OUTPUT_FILE,
                        mode="a",
                        header=not os.path.exists(OUTPUT_FILE),
                        index=False,
                        encoding='utf-8',
                        errors='ignore'
                    )
                else:
                    print(f"⚠️ Sin contenido: {paper['titulo'][:60]}")

                time.sleep(1)


print(f"\n✅ Total de papers útiles recopilados: {total_collected}")


# Comprobamos que no existan aarticulos extraidos sin contenido 
OUTPUT_FILE = "papers_guardados.csv"

if os.path.exists(OUTPUT_FILE):
    df_existente = pd.read_csv(OUTPUT_FILE)

    df_limpio = df_existente[df_existente['contenido'].notna() & (df_existente['contenido'] != '')]
    df_limpio.to_csv(OUTPUT_FILE, index=False)

    print(f"Archivo limpiado guardado.: {len(df_limpio)}")
