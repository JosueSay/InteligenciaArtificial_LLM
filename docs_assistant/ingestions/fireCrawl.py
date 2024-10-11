import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from firecrawl import FirecrawlApp
from langchain.schema import Document

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Inicializar los embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs2() -> None:
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    url = "https://starwars.fandom.com/wiki/Star_Wars:_Episode_I_The_Phantom_Menace"

    page_content = app.scrape_url(url=url,
                                  params={
                                      "onlyMainContent": True
                                  })

    print(page_content)
    doc = Document(page_content=str(page_content), metadata={"source": url})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200) # hay limite de memoria cuando se hacen embbedings por ello se hacen chunks
    docs = text_splitter.split_documents([doc])

    print(f"Going to add {len(docs)} to Pinecone")

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name=os.getenv("INDEX_NAME")
    )

    print(f"Loading {url} to Vector Store done")

# Punto de entrada principal para ejecutar la función 'ingest_docs'.
if __name__ == "__main__":
    ingest_docs2()
