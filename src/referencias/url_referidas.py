from utils import remove_sections,obtener_datos_unpaywall, extract_text_from_pdf ,extraer_info_relevante, find_direct_pdf_link, get_open_access_pdf, extract_dois_from_text, is_doi

import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PyPDF2 import PdfReader

import time
import pandas as pd
import re
import os
from dotenv import load_dotenv


def main():
    df = pd.read_csv("../segundo_semestre/SRI/proyecto final/papers_guardados.csv")
    urls_collected = []

    # limpiar contenido
    df['clean_contenido'] = df['contenido'].apply(lambda x: remove_sections(x, urls_collected))

    # guardar URLs encontradas
    urls_df = pd.DataFrame({'URL': urls_collected})
    urls_df.to_csv('urls_encontradas.csv', index=False)

    print(f"Se encontraron y guardaron {len(urls_collected)} URLs.")

    # cargar email de entorno
    load_dotenv()
    EMAIL = os.getenv("email")

    VISITED_DOIS_FILE = "dois_visitados.txt"
    OUTPUT_FILE = "../proyecto final/papers_guardados.csv"

    if os.path.exists(VISITED_DOIS_FILE):
        with open(VISITED_DOIS_FILE, "r") as f:
            existing_dois = set(f.read().splitlines())
    else:
        existing_dois = set()

    if os.path.exists(OUTPUT_FILE):
        df_existente = pd.read_csv(OUTPUT_FILE)
        dois_guardados = set(df_existente["doi"])
        results = df_existente.to_dict(orient="records")
    else:
        df_existente = pd.DataFrame()
        dois_guardados = set()
        results = []

    total_collected = 0

    for count, url in enumerate(urls_collected, start=1):
        print(count)

        if is_doi(url):
            doi = extract_dois_from_text(url)[0]
            if doi in dois_guardados:
                continue

            datos = obtener_datos_unpaywall(doi, EMAIL)
            if not datos:
                print("❌ Error en url extraída")
                continue

            paper = extraer_info_relevante(datos)
            if not paper:
                continue

            doi = paper["doi"]
            existing_dois.add(doi)

            landing_url = paper["landing_page"]
            pdf_url = get_open_access_pdf(doi, EMAIL)
            if not pdf_url:
                pdf_url = find_direct_pdf_link(landing_url)

            paper["pdf_url"] = pdf_url
            content = extract_text_from_pdf(pdf_url) if pdf_url else None
            paper["contenido"] = content

            if content and content != "No disponible":
                results.append(paper)
                dois_guardados.add(doi)
                with open(VISITED_DOIS_FILE, "a") as f_dois:
                    f_dois.write(doi + "\n")

                total_collected += 1

                print(f"✅ ({total_collected}) {paper['title'][:60]}...")

                pd.DataFrame([paper]).to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    header=not os.path.exists(OUTPUT_FILE),
                    index=False,
                    encoding='utf-8',
                    errors='ignore'
                )
            else:
                print(f"⚠️ Sin contenido: {paper['title'][:60]}")

            time.sleep(1)


if __name__ == "__main__":
    main()
