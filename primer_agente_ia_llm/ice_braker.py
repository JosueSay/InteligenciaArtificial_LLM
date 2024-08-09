# Este módulo está enfocado a leer una biografía y obtener datos sobre ella

from langchain.prompts import PromptTemplate  # Importa la clase PromptTemplate, utilizada para definir prompts que se envían a los modelos de lenguaje.
from langchain_openai import ChatOpenAI  # Importa la clase ChatOpenAI de la librería langchain_openai, que facilita la integración con el servicio de chat de OpenAI.
from dotenv import load_dotenv  # Importa la función load_dotenv de la librería dotenv, utilizada para cargar variables de entorno desde un archivo .env.
import os  # Importa el módulo os, que proporciona una forma de interactuar con el sistema operativo, incluyendo la gestión de variables de entorno.

information = """
Elon Reeve Musk (Pretoria, 28 de junio de 1971), conocido como Elon Musk, es un empresario, inversor y magnate sudafricano que también posee las nacionalidades canadiense y estadounidense. Es el fundador, consejero delegado e ingeniero en jefe de SpaceX; inversor ángel, director general y arquitecto de productos de Tesla, Inc.; fundador de The Boring Company; cofundador de Neuralink y OpenAI, aunque ya no tiene más participación en esta última por desacuerdos en el rumbo de la empresa. Además de ser el director de tecnología de X Corp..5​

Con un patrimonio neto estimado en unos 252.4 mil millones de dólares en julio de 2024,6​ llegando a ser la persona más rica del mundo según el índice de multimillonarios en tiempo real de Forbes.7​8​

Musk nació y se crío en una rica familia de Pretoria (Sudáfrica). Su madre es canadiense y su padre un sudafricano blanco. Estudió brevemente en la Universidad de Pretoria antes de trasladarse a Canadá a los 17 años. Se matriculó en la Universidad de Queen y se trasladó a la Universidad de Pensilvania dos años después, donde se graduó en Economía y Física. En 1995 se trasladó a California para asistir a la Universidad Stanford, pero en su lugar decidió seguir una carrera empresarial, cofundando la empresa de software web Zip2 con su hermano Kimbal. La empresa fue adquirida por Compaq por 307 millones de dólares en 1999. Ese mismo año, Musk cofundó el banco en línea X.com, que se fusionó con Confinity en 2000 para formar PayPal. La empresa fue comprada por eBay en 2002 por 1500 millones de dólares.

En 2002, Musk fundó SpaceX, fabricante aeroespacial y empresa de servicios de transporte espacial, de la que es CEO e ingeniero jefe. En 2003, se unió al fabricante de vehículos eléctricos Tesla Motors, Inc. (ahora Tesla, Inc.) como presidente y arquitecto de productos, convirtiéndose en su consejero delegado en 2008. En 2006, ayudó a crear SolarCity, una empresa de servicios de energía solar que posteriormente fue adquirida por Tesla y se convirtió en Tesla Energy. En 2015, cofundó OpenAI, una empresa de investigación sin ánimo de lucro que promueve la inteligencia artificial amigable. En 2016, cofundó Neuralink, una empresa de neurotecnología centrada en el desarrollo de interfaces cerebro-ordenador, y fundó The Boring Company, una empresa de construcción de túneles. También acordó la compra de la importante red social estadounidense Twitter en 2022 por 44 000 millones de dólares. Musk también ha propuesto el hyperloop. En noviembre de 2021, el director general de Tesla fue la primera persona de la historia en acumular una fortuna de 300 000 millones de dólares.9​

Ha sido criticado por hacer declaraciones poco científicas y controvertidas. En 2018, fue demandado por la Comisión de Bolsa y Valores de Estados Unidos (SEC) por tuitear falsamente que había conseguido financiación para una adquisición privada de Tesla. Llegó a un acuerdo con la SEC pero no admitió su culpabilidad, renunciando temporalmente a su presidencia y aceptando limitaciones en su uso de Twitter. En 2019, ganó un juicio por difamación presentado contra él por un espeleólogo británico que asesoró en el rescate de la cueva de Tham Luang. Musk también ha sido criticado por difundir información errónea sobre la pandemia de COVID-19 y teorías de conspiración; y por sus otras opiniones sobre asuntos como la inteligencia artificial, las criptomonedas y el transporte público.
"""

if __name__ == "__main__":
    # Cargar las variables de entorno
    load_dotenv()

    # Crear la plantilla del prompt
    summary_template = f"""
    Given the information {information} about a person, create:
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
    chain = summary_prompt_template | llm

    # Ejecutar la cadena con la información proporcionada
    res = chain.invoke(input={"information": information})

    # Imprimir el resultado
    print(res)
