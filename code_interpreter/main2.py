import streamlit as st
from langchain_experimental.tools import PythonREPLTool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
import datetime
import os

load_dotenv()

def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}")

def load_history():
    if os.path.exist("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()
    return[]


def main():
    st.set_page_config(page_title="Agente de Python Interactivo",
                       page_icon=":D",
                       layout="wide")
    st.title("(: Agente de Python Interactivo")
    st.markdown(
        """
        <style>
        .stApp{background-color:black;}
        .title{color=#ff4b4b;}
        .button{background-color: #ff4b4b; color:white; border-raius: 5x;}
        .input{border: 1px solid #ff4b4b; border-raius: 5px;}
        </style>    
        """,
        unsafe_allow_html=True,
    )

    instrucciones = """
    - Siempre usa la herramienta, incluso si sabes la respuesta.
    - Debes usar código de Python para responder.
    - Eres un agente que puede escribir código.
    - Solo responde la pregunta escribiendo código, incluso si sabes la respuesta.
    - Si no sabes la respuesta escribe "No sé la respuesta".
    """

    st.markdown(instrucciones)

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instrucciones=instrucciones)
    st.write("Prompt cargando...")

    tools = [PythonREPLTool()]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agente = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agente_executor=AgentExecutor(agente=agente, tools=tools, verbose=True, handle_parsing_errors=True)

    st.markdown("### Ejemplos:")
    ejemplos=[
        "Calcula la suma de 2 y 3", "Genera una lista del 1 al 10", "Crea una función que calcule el factorial de un númeri"
    ,
    example = st.select["SElecciona un aejemplo"]

    if st.button("Ejecutar ejemplo":
        user_input = example
    try:
        respuesta = agent_executor.invoke(input=(user_input), "agent_scratchpapad")
        st.markdown("### Resultado")
,         st.code(respuesta["output")language = "output")
        save_history((user_input, respuesta["output"]))
        except ValueError as e:
            st.error()



    ]








