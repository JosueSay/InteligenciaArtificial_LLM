import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain import hub
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

# Función para formatear el historial de la conversación
def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    formatted_history = "Conversation History:\n"
    for interaction in chat_history:
        role = interaction["role"]
        content = interaction["content"]
        if role == "human":
            formatted_history += f"User: {content}\n"
        else:
            formatted_history += f"Assistant: {content}\n"
    formatted_history += "End of History\n"
    return formatted_history


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    # Mostrar la consulta recibida
    print(f"\n\n=== NUEVA CONSULTA ===")
    print(f"Consulta recibida: {query}")

    # Mostrar el historial actual antes de procesar la consulta
    print("\nHistorial actual antes de la consulta:")
    if chat_history:
        for i, message in enumerate(chat_history, start=1):
            print(f"{i}. {message['role']}: {message['content']}")
    else:
        print("No hay historial previo.")

    # Crear embeddings de OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Conectar con el índice de Pinecone existente
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )

    # Crear el modelo LLM de OpenAI
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Formatear el historial de chat en un texto entendible para el modelo
    formatted_chat_history = format_chat_history(chat_history)
    print("\nHistorial formateado enviado al modelo:\n", formatted_chat_history)

    # Crear la cadena de preguntas y respuestas utilizando el método adecuado
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    # Ejecutar la consulta, incluyendo el historial formateado en el contexto
    result = qa_chain.invoke({"query": f"{formatted_chat_history}User: {query}"})

    # Mostrar la respuesta recibida del modelo
    print("\n=== RESPUESTA DEL MODELO ===")
    print(f"Respuesta del modelo: {result['result']}")
    print("\nDocumentos fuente:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'Fuente no disponible')}")

    # Agregar la interacción al historial como diccionarios
    chat_history.append({"role": "human", "content": query})
    chat_history.append({"role": "ai", "content": result['result']})

    # Mostrar el historial actualizado después de agregar la respuesta
    print("\nHistorial actualizado:")
    for i, message in enumerate(chat_history, start=1):
        print(f"{i}. {message['role']}: {message['content']}")

    # Estructurar los resultados para el frontend, incluyendo el historial actualizado
    new_result = {
        "query": query,
        "response": result["result"],
        "sources": [doc.metadata.get("source", "No source available") for doc in result["source_documents"]],
        "chat_history": chat_history  # Devolvemos el historial actualizado
    }

    return new_result


if __name__ == "__main__":
    chat_history = []
    print(run_llm(query="What is a Chain in LangChain?", chat_history=chat_history))
    print(run_llm(query="How can it be used for document retrieval?", chat_history=chat_history))
    print(run_llm(query="What are the key components of LangChain?", chat_history=chat_history))
    print(run_llm(query="How does LangChain integrate with LLMs?", chat_history=chat_history))
    print(run_llm(query="Can LangChain handle multiple documents at once?", chat_history=chat_history))
