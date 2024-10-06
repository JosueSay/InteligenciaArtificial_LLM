
# 🤖 Asistente de IA para Heroes de Overwatch

Un asistente que utiliza la API de embeddings de OpenAI con el modelo `text-embedding-3-small` para el tema de los héroes del juego *Overwatch 2*. Este proyecto emplea herramientas como **Pinecone** como índice y lugar de almacenamiento, creando una base vectorial a partir de información recolectada mediante web scraping, todo con una interfaz desarrollada en **Streamlit**.

## 🚀 Clonar el repositorio
Puedes clonar el repositorio usando el siguiente enlace:
```bash
git clone https://github.com/JosueSay/Selectivo_IA.git
```

## ⚙️ Configuración de la base vectorial en Pinecone

1. Accede a tu cuenta de [Pinecone](https://www.pinecone.io/).
2. Crea un índice:
    - **Nombre del Índice:** Especifica un nombre único.
    - **Dimensión del Índice:** Selecciona la dimensión usando `Setup by model` y escoge el modelo `text-embedding-3-small`.
    - **Proveedor de Nube:** Escoge **AWS** como proveedor.
        - **Región:** Selecciona la región, puede ser *us-east-1*, *us-west-2*, o *eu-west-1*.
    - **Crear el Índice:** Haz clic en "Crear".

## 📦 Instalar dependencias
Ejecuta el siguiente comando para instalar las dependencias necesarias:
```bash
pip install -r requirements.txt
```

## 🌍 Crear variables de entorno
Crea un archivo `.env` y añade lo siguiente:
```bash
OPENAI_API_KEY=         # Clave de OpenAI
PINECONE_API_KEY=       # Clave de Pinecone
INDEX_NAME=             # Nombre del índice de la base vectorial en Pinecone
PINECONE_ENVIRONMENT=   # Región del índice 
```

## 🗂️ Archivos importantes

- **`.\backend\core_langchain_docs.py`**: Script encargado de hacer backend para realizar solicitudes a la API de OpenAI usando el embebido `text-embedding-3-small`.
  
- **`.\frontend\app_docs.py`**: Script encargado de montar el frontend usando Streamlit.
  
- **`.\scrappers\over_scrapp.py`**: Script encargado de hacer web scraping y obtener datos para montarlos en una base vectorial en Pinecone.

## ▶️ Ejecutar el código

1. Primero, ejecuta el script para el web scraping de la página de Overwatch:
   ```bash
   python .\scrappers\over_scrapp.py
   ```

2. Luego, ejecuta el script para la interfaz:
   ```bash
   streamlit run .\frontend\app_docs.py
   ```

¡Ingresa tu prompt para probar el chat! 💬

## 🎥 Video del funcionamiento
Puedes ver el video del funcionamiento en el siguiente enlace: [Video de funcionamiento](https://youtu.be/uDdx9rBryYM)
