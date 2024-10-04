import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeVectorStore

load_dotenv()

# Inicializar Pinecone con la nueva clase y ServerlessSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Crear el índice si no existe
if os.getenv("INDEX_NAME") not in pc.list_indexes().names():
    pc.create_index(
        name=os.getenv("INDEX_NAME"),
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=os.getenv("PINECONE_ENVIRONMENT")
        )
    )

def run_llm(query: str) -> Any:
    # Crear objeto de embeddings de OpenAI (usará la clave desde OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Crear objeto de búsqueda basado en Pinecone
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("INDEX_NAME"), embedding=embeddings
    )

    # Crear modelo LLM de OpenAI (usará la clave desde OPENAI_API_KEY)
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Crear la cadena de preguntas y respuestas con la búsqueda
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    return qa.invoke({"query": query})

if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
