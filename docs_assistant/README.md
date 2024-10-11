
# 🤖 Asistente de IA para Langchain

Un asistente que utiliza la API de embeddings de OpenAI con el modelo `text-embedding-3-small` para la documentación local de archivos html sobre Langchain. Este proyecto emplea herramientas como **Pinecone** como índice y lugar de almacenamiento, creando una base vectorial a partir de información recolectada mediante web scraping y una interfaz desarrollada en **Streamlit**.

## 🚀 Clonar el repositorio
Puedes clonar el repositorio usando el siguiente enlace y cambiar a la rama `LAB5_DOC_ASSISTANT`:
```bash
git clone https://github.com/JosueSay/Selectivo_IA.git
git checkout LAB5_DOC_ASSISTANT
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
  
- **`.\ingestions\ingestion.py`**: Script encargado de hacer la ingestion de data al indice de la base vectorial en Pinecone.

## ▶️ Ejecutar el código

1. Primero, ejecuta el script para el llenar la base vectorial con la data de los archivos html en `data/langchain-docs`:
   ```bash
   python .\ingestions\ingestion.py
   ```

2. Luego, ejecuta el script para la interfaz:
   ```bash
   streamlit run .\frontend\app_docs.py
   ```

¡Ingresa tu prompt para probar el chat! 💬

## 🎥 Video del funcionamiento
Puedes ver el video del funcionamiento en el siguiente enlace: [Video de funcionamiento](https://youtu.be/55dGOiw7cbI?si=gf0IQJ3UD7SaeaOB)
