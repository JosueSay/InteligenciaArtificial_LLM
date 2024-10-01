import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

# Cargar variables de entorno
load_dotenv()

if __name__ == '__main__':
    print("Carga del archivo PDF...")

    # Cargar el pdf
    pdf_path = "D:\\UVG GitHub Repositorios\\Selectivo_IA\\local_vector_database\\2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    print("Dividir el PDF en chunks...")

    # Separar el documento en chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    print("Guardar los chunks en una base vectorial local...")

    # Crear el objeto embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Guardar los embeddings localmente
    vectorstore.save_local("faiss_index_react")

    print("Cargar embedding de la base vectorial local...")

    # Cargar el índice FAISS guardado
    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    print("Creación y ejecución del agente...")

    # Crear la cadena de recuperación
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(),
        retrieval_qa_chat_prompt,
    )

    # Crear la cadena de recuperación
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(),
        combine_docs_chain,
    )

    print("Consulta...")

    # Ejecutar la consulta
    res = retrieval_chain.invoke(
        {"input": "Give me the gist of ReAct in 3 sentences"}
    )

    print(res["answer"])
    print("End of process...")
