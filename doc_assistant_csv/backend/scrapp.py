import os
import time
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Inicializar los embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# URL base
base_url = "https://overwatch.blizzard.com/es-es/heroes/"


# Función para realizar el scraping de todos los héroes y enviar a Pinecone
def scrape_all_heroes_and_ingest():
    # Realizar la solicitud para obtener la página principal
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Encontrar todos los héroes en la página
    heroes = soup.find_all('blz-hero-card')

    if not heroes:
        print("No se encontraron héroes.")
        return

    for hero in heroes:
        hero_name = hero['hero-name']
        hero_role = hero['data-role']
        hero_url = "https://overwatch.blizzard.com" + hero['href']

        # Obtener la historia de cada héroe
        hero_response = requests.get(hero_url)
        hero_soup = BeautifulSoup(hero_response.content, 'html.parser')

        # Obtener la historia
        historia_tag = hero_soup.find('p', slot='description')
        historia = historia_tag.text if historia_tag else "Historia no disponible."

        # Obtener habilidades
        habilidades = hero_soup.find_all('blz-feature')
        habilidades_info = []

        for habilidad in habilidades:
            habilidad_nombre = habilidad.find('h3', slot='heading').text
            habilidad_descripcion = habilidad.find('p').text
            habilidades_info.append(f"Habilidad: {habilidad_nombre} - Descripción: {habilidad_descripcion}")

        # Crear el contenido del documento
        contenido = f"Héroe: {hero_name}\nRol: {hero_role}\nURL: {hero_url}\nHistoria: {historia}\n" + "\n".join(
            habilidades_info)

        # Crear un documento con el contenido extraído usando la clase Document
        document = Document(
            page_content=contenido,
            metadata={"hero_name": hero_name, "role": hero_role, "source": hero_url}
        )

        # Mostrar información del héroe
        print(f"Héroe: {hero_name}, Rol: {hero_role}, URL: {hero_url}")
        print(f"Historia de {hero_name}: {historia}")
        print("\n".join(habilidades_info))

        # Pausar para evitar hacer muchas solicitudes en poco tiempo
        time.sleep(1)

        # Crear el text_splitter para dividir el documento en chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        split_documents = text_splitter.split_documents([document])

        print(f"Split en {len(split_documents)} documentos. Enviando a Pinecone...")

        # Enviar el documento procesado a Pinecone
        PineconeVectorStore.from_documents(
            split_documents, embedding=embeddings, index_name=os.getenv("INDEX_NAME")
        )

    print("Scraping y envío a Pinecone completados.")


# Punto de entrada principal para ejecutar la función 'scrape_all_heroes_and_ingest'.
if __name__ == "__main__":
    scrape_all_heroes_and_ingest()
