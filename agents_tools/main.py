# Este módulo está enfocado a leer una biografía y obtener datos sobre ella
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate  # Importa la clase PromptTemplate, utilizada para definir prompts que se envían a los modelos de lenguaje.
from langchain_openai import ChatOpenAI  # Importa la clase ChatOpenAI de la librería langchain_openai, que facilita la integración con el servicio de chat de OpenAI.
from dotenv import load_dotenv  # Importa la función load_dotenv de la librería dotenv, utilizada para cargar variables de entorno desde un archivo .env.
import os  # Importa el módulo os, que proporciona una forma de interactuar con el sistema operativo, incluyendo la gestión de variables de entorno.

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    # Cargar las variables de entorno
    load_dotenv()

    # Crear la plantilla del prompt
    summary_template = """
    Given the linkedin information {information} about a person, create:
    1. A short summary
    2. Two interesting facts about them
    """

    # Crear el objeto PromptTemplate para enviar a la API
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Crear el objeto LLM a usar, temperature es de acuerdo a la creatividad de respuesta
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Crear la secuencia de ejecución
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/roger-d%C3%ADaz-0946758b/",)

    # Ejecutar la cadena con la información proporcionada
    res = chain.invoke(input={"information": linkedin_data})

    # Imprimir el resultado
    print(res)

