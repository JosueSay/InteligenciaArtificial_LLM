import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
base_url = 'https://myanimelist.net/anime.php'

# Función para realizar el scraping de todos los géneros y animes, y luego enviarlos a Pinecone
def scrape_genres_and_animes():
    response = requests.get(base_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Buscamos todos los divs que tienen la clase "genre-list"
        genre_divs = soup.find_all('div', class_='genre-list al')

        for genre_div in genre_divs:
            genre_link = genre_div.find('a', class_='genre-name-link')
            if genre_link:
                genre_url = genre_link['href']
                genre_id = genre_url.split('/')[3]
                genre_name = genre_link.text.strip()  # Nombre y conteo de animes

                print(f"\nGénero ID: {genre_id}, Nombre: {genre_name}")
                scrape_animes_by_genre(genre_id) # animes del mismo genero
    else:
        print(f"Error al acceder a la página: {response.status_code}")


# Función para obtener los animes por género y enviarlos a Pinecone
def scrape_animes_by_genre(genre_id):
    genre_url = f'https://myanimelist.net/anime/genre/{genre_id}'
    response = requests.get(genre_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Buscamos todos los animes bajo el género
        anime_divs = soup.find_all('div', class_='js-anime-category-producer')

        for anime_div in anime_divs:
            anime_title_tag = anime_div.find('a', class_='link-title')

            if anime_title_tag:
                anime_url = anime_title_tag['href']
                anime_id = anime_url.split('/')[4]
                anime_name = anime_title_tag.text.strip()

                print(f"\nAnime ID: {anime_id}, Nombre: {anime_name}")
                # Obtenemos la información del anime
                document = scrape_anime_information(anime_url)
                if document:
                    # Pausar para evitar hacer muchas solicitudes en poco tiempo
                    time.sleep(0.5)

                    # Crear el text_splitter para dividir el documento en chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
                    split_documents = text_splitter.split_documents([document])

                    print(f"Dividido en {len(split_documents)} documentos. Enviando a Pinecone...")

                    # Enviar el documento procesado a Pinecone
                    PineconeVectorStore.from_documents(
                        split_documents, embedding=embeddings, index_name=os.getenv("INDEX_NAME")
                    )
    else:
        print(f"Error al acceder a la página del género: {response.status_code}")

# Función para extraer y preparar la información del anime
def scrape_anime_information(anime_url):
    response = requests.get(anime_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Diccionario para almacenar la información del anime
        anime_info = {}

        # Buscamos los campos con la clase "spaceit_pad"
        info_divs = soup.find_all('div', class_='spaceit_pad')

        for div in info_divs:
            label = div.find('span', class_='dark_text')
            if label:
                label_text = label.text.strip().replace(":", "")  # Removemos ":" del final de las etiquetas

                # Obtenemos el valor asociado
                if label_text in ["Producers", "Licensors", "Studios", "Genres", "Themes", "Demographic"]:
                    value = ', '.join([a.text for a in div.find_all('a')])
                else:
                    value = div.get_text(strip=True).replace(label.text.strip(), "").strip()

                anime_info[label_text] = value

        # Crear un documento con el contenido extraído usando la clase Document
        contenido = "\n".join([f"{key}: {value}" for key, value in anime_info.items()])
        document = Document(
            page_content=contenido,
            metadata={"anime_url": anime_url}
        )

        return document
    else:
        print(f"Error al acceder a la página del anime: {response.status_code}")
        return None

if __name__ == "__main__":
    scrape_genres_and_animes()
