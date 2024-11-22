from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from pinecone import Pinecone, ServerlessSpec
from typing import Any, Dict, List, Tuple

# Cargar variables de entorno
load_dotenv()

# Inicializar Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
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

# Cargar el prompt base
base_prompt = hub.pull("langchain-ai/react-agent-template").partial(
    instructions="""
    You are an intelligent assistant that always responds in the same language as the input query.
    If the user asks a question in Spanish, respond in Spanish. If the user asks in English, respond in English.
    Maintain language consistency throughout the conversation.
    """
)

# Variable global para rastrear la última herramienta utilizada
last_tool_used = None

def wrap_tool(tool, name):
    """
    Envuelve la función de una herramienta para registrar su uso.
    """
    def wrapped_func(*args, **kwargs):
        global last_tool_used
        last_tool_used = name  # Registrar el nombre de la herramienta
        return tool(*args, **kwargs)
    return wrapped_func

# Crear herramientas para Pinecone y CSV
def create_tools():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )

    # Función de consulta principal
    def pinecone_query(query: str) -> Dict[str, Any]:
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(verbose=True, temperature=0),
            retriever=docsearch.as_retriever(),
            return_source_documents=True
        )
        response = qa_chain.invoke({"query": query})

        # Extraer metadata de los documentos fuente
        source_documents = response.get("source_documents", [])
        metadata = [
            doc.metadata.get("source", "No source available")
            for doc in source_documents
        ]

        return {
            "response": response.get("result", ""),
            "metadata": metadata
        }

    # Herramienta para obtener respuestas
    def get_response(query: str) -> str:
        result = pinecone_query(query)
        return result["response"]

    # Herramienta para obtener metadata
    def get_metadata(query: str) -> List[str]:
        result = pinecone_query(query)
        return result["metadata"]

    # Definición de herramientas con wrapper
    response_tool = Tool(
        name="Pinecone Response",
        func=wrap_tool(get_response, "Pinecone Response"),
        description="Use this tool to get answers for general Overwatch lore or hero abilities questions."
    )

    metadata_tool = Tool(
        name="Pinecone Metadata",
        func=wrap_tool(get_metadata, "Pinecone Metadata"),
        description="Use this tool to get metadata (sources) for Overwatch lore or hero abilities questions."
    )

    # CSV Tool
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "hero_stats.csv"))
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(verbose=True, temperature=0),
        path=csv_path,
        verbose=True,
        allow_dangerous_code=True
    )
    csv_tool = Tool(
        name="CSV Agent",
        func=wrap_tool(csv_agent.invoke, "CSV Agent"),
        description="Use this tool for detailed hero stats from the CSV dataset."
    )

    python_repl_tool = PythonREPLTool()

    return [response_tool, metadata_tool, csv_tool, python_repl_tool]

# Función principal para ejecutar la consulta
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    global last_tool_used
    last_tool_used = None  # Resetear antes de cada consulta
    sources = None # metadata
    tools = create_tools()
    grand_agent = create_react_agent(
        prompt=base_prompt,
        llm=ChatOpenAI(temperature=0),
        tools=tools,
    )
    executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    # Ejecutar consulta con el agente principal y manejar posibles errores
    try:
        result = executor.invoke({"input": query, "chat_history": chat_history, "handle_parsing_errors": True})
    except Exception as e:
        return {"query": query, "response": "I don't know", "chat_history": chat_history}

    # Formatear historial correctamente
    chat_history.append({"role": "human", "content": query})
    chat_history.append({"role": "ai", "content": result["output"]})


    # Si la herramienta utilizada fue la del CSV, no generar metadata
    if last_tool_used == "Pinecone Response":
        # Generar metadata si no fue CSV Agent
        sources = tools[1].func(query)

    for i, message in enumerate(chat_history, start=1):
        print(f"\t{i}. {message['role']}: {message['content']}")

    # Estructurar respuesta
    return {
        "query": query,
        "response": result["output"],
        "sources": sources,
        "chat_history": chat_history,
    }

# Pruebas iniciales
if __name__ == "__main__":
    chat_history = []

    # Prompt 1: Rol de Moira
    response_1 = run_llm(query="¿Cuál es el rol de Moira en Overwatch?", chat_history=chat_history)
    print(response_1)

    # Prompt 2: Winrate por rango de Moira
    response_2 = run_llm(query="Dime el winrate para cada rango del héroe Moira", chat_history=chat_history)
    print(response_2)

    # Prompt 3: KDA por rango de Moira
    response_3 = run_llm(query="Ahora su KDA para cada rango", chat_history=chat_history)
    print(response_3)

    # Prompt 4: Habilidades de Reinhardt
    response_4 = run_llm(query="¿Qué habilidades tiene Reinhardt en Overwatch?", chat_history=chat_history)
    print(response_4)

    # Prompt 5: Winrate de D.Va por rango
    response_5 = run_llm(query="Muéstrame el winrate de D.Va en cada rango", chat_history=chat_history)
    print(response_5)
