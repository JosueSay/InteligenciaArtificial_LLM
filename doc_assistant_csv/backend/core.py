from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from pinecone import Pinecone, ServerlessSpec
from typing import Any, Dict, List

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

# Cargar el prompt base (cacheado para eficiencia)
base_prompt = hub.pull("langchain-ai/react-agent-template")

# Crear herramientas para Pinecone y CSV
def create_tools():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )

    # Pinecone Tool
    def pinecone_query(query: str):
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(verbose=True, temperature=0),
            retriever=docsearch.as_retriever(),
            return_source_documents=True
        )
        return qa_chain.invoke({"query": query})

    pinecone_tool = Tool(
        name="Pinecone Agent",
        func=pinecone_query,
        description="Use this tool for general Overwatch lore or hero abilities questions."
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
        func=csv_agent.invoke,
        description="Use this tool for detailed hero stats from the CSV dataset."
    )

    return [pinecone_tool, csv_tool]

# Función principal para ejecutar la consulta
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    tools = create_tools()
    grand_agent = create_react_agent(
        prompt=base_prompt.partial(instructions=""),
        llm=ChatOpenAI(temperature=0),
        tools=tools,
    )
    executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    # Ejecutar consulta con el agente principal
    result = executor.invoke({"input": query})

    # Formatear historial
    chat_history.append({"role": "human", "content": query})
    chat_history.append({"role": "ai", "content": result["output"]})

    # Estructurar respuesta
    return {
        "query": query,
        "response": result["output"],
        "sources": result.get("source_documents", []),
        "chat_history": chat_history,
    }

# Pruebas iniciales
if __name__ == "__main__":
    chat_history = []
    print(run_llm(query="¿Cuál es el rol de Moira en Overwatch?", chat_history=chat_history))
    print(run_llm(query="¿Cuáles son las habilidades de Moira?", chat_history=chat_history))
    print(run_llm(query="¿Cuál es el KDA de Ana en rango Diamante?", chat_history=chat_history))
