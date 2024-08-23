import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter # Esto hace que podamos medir la cantidad de chunks por caracteres
from langchain_openai import OpenAIEmbeddings # son los chunks donde se guarda la información
from langchain_pinecone import PineconeVectorStore # conexión con pinecone

load_dotenv()

if __name__=='__main__':
    print("Ingesting...")
    loader = TextLoader("D:\\UVG GitHub Repositorios\\Selectivo_IA\\agent_vectorial_base\\mediumblog1.txt", encoding='utf-8') # pide dirección del archivo a cargar
    document = loader.load()
    print("Splitting...")

    # Creación de objetos para los chunks para meter infor en el base vectorial
    # Chunkoverlap realiza un método de partición de párrafos que tengan parte del párrafo anterior en el nuevo
    # No se necesita en esto, por eso se coloca 0
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document) # se parte esos textos del documento (un documento que se está partiendo obteniendo varios texts)
    print(f"{len(texts)} chunks were created.")

    # Crear el obtjeto embeding para pasar lo partido antes a la base vectorial
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])