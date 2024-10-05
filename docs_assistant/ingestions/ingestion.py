import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Inicializar los embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Función para cargar y extraer texto de archivos HTML desde todas las subcarpetas
def loadDocumentHTML(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".html"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    # Usar BeautifulSoup para parsear el HTML
                    soup = BeautifulSoup(file, "html.parser")
                    text = soup.get_text(separator="\n")  # Extraer todo el texto del HTML

                    # Crear un documento con el contenido extraído usando la clase Document
                    document = Document(
                        page_content=text,
                        metadata={"source": filepath}
                    )
                    documents.append(document)
    return documents


# Función para dividir los documentos en chunks y enviarlos a Pinecone
def ingest_docs():
    # Obtener el directorio actual de donde está este archivo ingestion.py
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Subir al nivel del proyecto (docs_assistant) y acceder a la carpeta "data"
    project_directory = os.path.abspath(os.path.join(current_directory, ".."))

    # Construir la ruta relativa desde la carpeta del proyecto
    directory = os.path.join(project_directory, "data", "langchain-docs", "api.python.langchain.com", "en", "latest")

    raw_documents = loadDocumentHTML(directory)

    print(f"loaded {len(raw_documents)} documents")

    # Crear el text_splitter para dividir los documentos en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    print(f"split {len(documents)} documents")

    for doc in documents:
        # Obtener la ruta del archivo original desde los metadatos
        new_url = doc.metadata["source"]

        # Reemplazar correctamente la ruta local completa hasta 'api.python.langchain.com' por la URL base
        new_url = new_url.replace(os.path.join(project_directory, "data", "langchain-docs", "api.python.langchain.com"),
                                  "https://api.python.langchain.com")

        # Asegurarse de que las barras invertidas se conviertan en barras inclinadas
        new_url = new_url.replace("\\", "/")

        # Actualizar la metadata con la nueva URL correctamente formateada
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} documents to Pinecone")

    # Enviar los documentos procesados a Pinecone
    PineconeVectorStore.from_documents(
        documents, embedding=embeddings, index_name=os.getenv("INDEX_NAME")
    )


# Punto de entrada principal para ejecutar la función 'ingest_docs'.
if __name__ == "__main__":
    ingest_docs()
