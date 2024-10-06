import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

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

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    # Mostrar la consulta recibida y el historial
    # print(f"(BACKEND) Consulta recibida: {query}")
    # print(f"(BACKEND) Historial recibido:")
    # if chat_history:
    #     for i, message in enumerate(chat_history, start=1):
    #         print(f"\t{i}. {message['role']}: {message['content']}")
    # else:
    #     print("\tNo hay historial previo.")

    # Crear embeddings de OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Conectar con el índice de Pinecone existente
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )

    # Crear el modelo LLM de OpenAI
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Crear la cadena de preguntas y respuestas utilizando el método adecuado
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    # Ejecutar la consulta, incluyendo el historial formateado en el contexto
    if not chat_history:
        # Construir la consulta sin historial
        result = qa_chain.invoke({"query": f"User: {query}"})
    else:
        # Construir la consulta con el historial
        result = qa_chain.invoke({"query": f"{chat_history}User: {query}"})

    # Mostrar la respuesta recibida del modelo
    # print(f"(BACKEND) Respuesta de IA: {result['result']}")
    # print("(BACKEND) Fuentes de de IA:")
    # for doc in result["source_documents"]:
    #     print(f"\t- {doc.metadata.get('source', 'Fuente no disponible')}")

    # Agregar la interacción al historial como diccionarios
    chat_history.append({"role": "human", "content": query})
    chat_history.append({"role": "ai", "content": result['result']})

    # Mostrar el historial actualizado después de agregar la respuesta
    # print("(BACKEND)Historial actualizado:")
    # for i, message in enumerate(chat_history, start=1):
    #     print(f"\t{i}. {message['role']}: {message['content']}")

    # Estructurar los resultados para el frontend, incluyendo el historial actualizado
    new_result = {
        "query": query,
        "response": result["result"],
        "sources": [doc.metadata.get("source", "No source available") for doc in result["source_documents"]],
        "chat_history": chat_history
    }

    return new_result


if __name__ == "__main__":
    chat_history = []
    print(run_llm(query="What is a Chain in LangChain?", chat_history=chat_history))
    print(run_llm(query="How can it be used for document retrieval?", chat_history=chat_history))
    print(run_llm(query="What are the key components of LangChain?", chat_history=chat_history))
    print(run_llm(query="How does LangChain integrate with LLMs?", chat_history=chat_history))
    print(run_llm(query="Can LangChain handle multiple documents at once?", chat_history=chat_history))
