# Acá se crea el agente y se corre

import os
from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# Libreria para utilizar tools, funciones que comunican
# al LLM con el exterior
from langchain_core.tools import Tool
# Importar los archivos necesarios para crear un agente
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)

# Sirve para obtener prompts por la comunidad
from langchain import hub

load_dotenv()

def lookup(name: str, ) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    #Crear el template
    template = """given the full name {name_of_person} I want you to get me a link to their LinkedIn profile page.
    Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful when you need to get a LinkedIn page URL",
        ) # De que es esa tool
    ]
    #hwchase es el perfil del creador de langchain, se obtiene el motor de razonamiento
    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # Visualizar el proceso de razonamiento -> verbose
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    # Correrlo
    result = agent_executor.invoke(
        input={"input": prompt_template.format(name_of_person=name)}
    )

    linkedin_profile_url = result["output"]

    return linkedin_profile_url

if __name__ == "__main__":
    linkedin_url = lookup(name="Josué Say Universidad del Valle de Guatemala")
    print(linkedin_url)