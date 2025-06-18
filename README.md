# 📚 Proyecto de Sistemas de Recuperación de Información sobre Investigación Científica

**Retrieve-Augmented Generation (RAG) for Scientific Research**

Este proyecto implementa un chatbot con recuperación de información y generación aumentada por recuperación (RAG), especializado en artículos científicos. El sistema permite realizar consultas en lenguaje natural y obtener respuestas relevantes basadas en contenido previamente indexado.

---

## ✍️ Autores

- Jennifer Sánchez – [@Jennyfer2004](https://github.com/Jennyfer2004)
- Reinaldo Canvas – [@Reinicg](https://github.com/Reinicg)

---

## 🧩 Descripción del Problema

El objetivo principal del proyecto es facilitar la recuperación de información relevante a partir de artículos científicos. Para ello, se emplean técnicas de embeddings semánticos y búsqueda vectorial, que permiten encontrar documentos relacionados con una consulta dada. Además, se implementa un chatbot que consulta la base de datos vectorial usando lenguaje natural, haciendo la interacción más intuitiva.

---

## ⚙️ Requerimientos

- Tener una cuenta de correo válida (Gmail) para acceder a la API de Unpaywall.
- Obtener un API token desde [Hugging Face](https://huggingface.co/settings/tokens).
- Tener Python >= 3.9 y `pip` instalado.
- Instalar las dependencias listadas en `requirements.txt`.

---

## 🔌 APIs Utilizadas

- **[Hugging Face Inference API](https://huggingface.co/inference-api)**  
  Utilizada para generar respuestas mediante el modelo de lenguaje `HuggingFaceH4/zephyr-7b-beta`, accesible a través de peticiones autenticadas.

- **[Crossref REST API](https://api.crossref.org/works)**  
  Permite recuperar metadatos de artículos científicos, como título, autores, DOI y fecha de publicación, mediante búsquedas por palabra clave.

- **[Unpaywall API](https://unpaywall.org/products/api)**  
  Se emplea para obtener enlaces a versiones en acceso abierto de artículos científicos, utilizando el DOI y un correo electrónico válido.

---

## 🚀 Ejecución del Proyecto

1. Clona el repositorio:
   ```bash
   git clone https://github.com/[tu_usuario]/[nombre_del_repositorio].git
   cd [nombre_del_repositorio]
2.Instala las dependencias:
    ```bash
    pip install -r requirements.txt
3.Ejecuta el script de inicio
    ```bash
    ./startup.sh
