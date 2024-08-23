import json
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

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

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/josuesay/",)

    # Ejecutar la cadena con la información proporcionada
    res = chain.invoke(input={"information": linkedin_data})

    # Guardar el resultado en un archivo JSON
    with open('respuesta.json', 'w') as file:
        json.dump(res, file, indent=4)

    # Imprimir el resultado
    print(res)
