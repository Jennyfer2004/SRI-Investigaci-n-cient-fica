import os
import pandas as pd
import time

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from preprocesamiento import clean_text, remove_sections, normalize_text

OUTPUT_FILE = "/home/jennifer/Descargas/papers_guardados-2.csv"

def main():
    if os.path.exists(OUTPUT_FILE):
        df_existente = pd.read_csv(OUTPUT_FILE)
        df_limpio = df_existente[df_existente['contenido'].notna() & (df_existente['contenido'] != '')]
        df_limpio.to_csv(OUTPUT_FILE, index=False)

        print(f"Archivo limpiado guardado.: {len(df_limpio)}")
        print(f"Archivo limpiado guardado.: {len(df_existente)}")

    df = pd.read_csv('/home/jennifer/Documentos/tercer_año/segundo_semestre/SRI/proyecto final/papers_guardados.csv')
    df1 = pd.read_csv(OUTPUT_FILE)

    # Concatenar df y df1
    df = pd.concat([df, df1], axis=0, ignore_index=True)

    df['clean_abstract'] = df['abstract'].apply(clean_text)
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_contenido'] = df['contenido'].apply(remove_sections)
    df['clean_contenido'] = df['clean_contenido'].apply(normalize_text)
    df['clean_contenido'] = df['clean_contenido'].apply(clean_text)
    df = df[df['clean_contenido'].str.len() > 200]

    print(f"[{time.strftime('%H:%M:%S')}] INFO: Convirtiendo DataFrame a Documentos LangChain...")
    documents = []
    for _, row in df.iterrows():
        content = f"{row['clean_abstract']}\n\n{row['clean_contenido']}"
        metadata = {
            'titulo': row['clean_title'],
            'doi': row['doi'],
            'autores': row['authors'],
            'publicado': row['published'],
            'idioma': row['language'],
            'url_pdf': row['pdf_url'],
            'url_landing': row['landing_page'],
        }
        documents.append(Document(page_content=content, metadata=metadata))
    print(f"[{time.strftime('%H:%M:%S')}] INFO: {len(documents)} documentos creados.")

    print(f"[{time.strftime('%H:%M:%S')}] INFO: Iniciando división de documentos en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", r"(?<=\. )", " "],
        length_function=len,
        is_separator_regex=True
    )

    chunks = []
    for i, doc in enumerate(documents):
        splits = text_splitter.split_documents([doc])
        for j, split in enumerate(splits):
            split.metadata['fragmento_id'] = f"{doc.metadata['doi']}_part{j+1}"
        chunks.extend(splits)
        print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Documento {i+1} dividido en {len(splits)} fragmentos.")

    print(f"[{time.strftime('%H:%M:%S')}] INFO: Total de {len(chunks)} fragmentos creados.")

    print(f"[{time.strftime('%H:%M:%S')}] INFO: Cargando modelo de embeddings HuggingFace...")
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"[{time.strftime('%H:%M:%S')}] INFO: Modelo de embeddings cargado correctamente.")

    print(f"[{time.strftime('%H:%M:%S')}] INFO: Creando base vectorial con ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory="./repositorio_vectorial",
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"[{time.strftime('%H:%M:%S')}] INFO: Guardado base vectorial en disco...")

if __name__ == "__main__":
    main()
