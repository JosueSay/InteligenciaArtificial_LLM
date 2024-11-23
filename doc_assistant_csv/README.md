# 🤖 Asistente de IA para Héroes de Overwatch

Este proyecto es un asistente inteligente que utiliza la API de embeddings de OpenAI (`text-embedding-3-small`) para proporcionar información sobre los héroes del juego *Overwatch 2*. Permite realizar consultas sobre estadísticas, habilidades y roles de los héroes, utilizando herramientas como **Pinecone** para manejar una base de datos vectorial y un archivo CSV; interactuando mediante una interfaz en streamlit.

El asistente tiene la capacidad de **decidir dinámicamente** qué herramienta usar (Pinecone o CSV) dependiendo de la consulta, y mantiene un historial para gestionar el contexto de la conversación.

---

## 🗂️ Estructura del proyecto

```plaintext
doc_assistant_csv/
├── .idea/                     # Archivos de configuración del entorno
├── .venv/                     # Entorno virtual para Python
├── backend/                   # Lógica principal del backend
│   ├── core.py                # Funciones principales para interactuar con el modelo LLM
│   └── scrapp.py              # Script para realizar web scraping
├── docs/                      # Carpeta con los datos csv
│   └── hero_stats.csv         # Datos de estadísticas de héroes
├── frontend/                  # Interfaz gráfica construida con Streamlit
│   └── app.py                 # Script principal del frontend
├── .env                       # Variables de entorno
├── README.md                  # Archivo de documentación del proyecto
├── requirements.txt           # Lista de dependencias
```

---

## 🚀 Clonar el repositorio

```bash
git clone https://github.com/JosueSay/Selectivo_IA.git
git checkout PROY2_LLM_SELECTION_CSV_PINECONE 
```

---

## ⚙️ Configuración de Pinecone

1. Accede a tu cuenta de [Pinecone](https://www.pinecone.io/).
2. Configura un índice:
   - **Nombre del Índice:** Crea un nombre único.
   - **Dimensión del Índice:** Usa `Setup by model` y selecciona el modelo `text-embedding-3-small`.
   - **Proveedor de Nube:** Escoge **AWS**.
       - **Región:** Selecciona *us-east-1*, *us-west-2*, o *eu-west-1*.
   - **Crear el Índice:** Haz clic en "Crear".

---

## 📦 Instalación de dependencias

Ejecuta el siguiente comando para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

---

## 🌍 Crear variables de entorno

Crea un archivo `.env` con las siguientes claves:

```bash
OPENAI_API_KEY=         # Tu clave de OpenAI
PINECONE_API_KEY=       # Tu clave de Pinecone
INDEX_NAME=             # Nombre del índice de Pinecone
PINECONE_ENVIRONMENT=   # Región del índice de Pinecone
```

---

## ▶️ Ejecución

1. Navega a la carpeta del proyecto:

   ```bash
   cd doc_assistant_csv
   ```

2. Ejecuta el script para recolectar datos con web scraping y llenado de la base vectorial en Pinecone:

   ```bash
   python backend/scrapp.py
   ```

3. Ejecuta la interfaz gráfica:

   ```bash
   streamlit run frontend/app.py
   ```

Se abrirá una pestaña en tu navegador y si no lo hace, accede al enlace local proporcionado por Streamlit y empieza a interactuar con el asistente.

---

## 💡 Prompts Recomendados

Prueba los siguientes prompts para evaluar el comportamiento del asistente:

 
1. **¿Cuál es el rol de Moira en Overwatch?**
2. **Dime el winrate para cada rango del héroe Moira.**
3. **Ahora su KDA para cada rango.**
4. **¿Qué habilidades tiene Reinhardt en Overwatch?**
5. **Muéstrame el winrate de D.Va en cada rango.**

También puedes utilizar los prompts por default en el **`menú desplegable`** y enviar el prompt para  probar la funcionalidad del agente.

---

## 🎥 Videos
- [**Video Configuración Base**](https://youtu.be/3ZOYn6ikhnw)
- [**Video Demostrativo**](https://youtu.be/meVhIajTNwA)
