import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.core import run_llm

# Configuración global de la página
st.set_page_config(
    page_title="ChatBot Assistant",  # Título de la página
    page_icon="🤖",  # Favicon (puedes usar emojis o un archivo .ico)
    layout="wide",  # Opcional: 'wide' para pantalla ancha, 'centered' para centrado
    initial_sidebar_state="collapsed",  # Opcional: 'collapsed' o 'expanded'
)

st.title("ChatBot Assistant")

# Historial que se obtiene después de cada prompt con LLM
if "chat_local" not in st.session_state:
    st.session_state.chat_local = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Define las solicitudes predefinidas
requests = [
    "Dime la información general de los héroes de OW",
    "Dime la información general del CSV de los héroes de OW",
    "Dame los datos del CSV de un héroe de OW aleatorio"
]

# Crear un menú desplegable para seleccionar la solicitud
selected_request = st.selectbox("Selecciona una solicitud de trabajo para el agente:", requests)

# Botón para enviar la solicitud seleccionada
if st.button("Enviar solicitud"):
    # Procesar la solicitud seleccionada
    with st.spinner("Cargando respuesta..."):
        # Llamar a la función LLM con el input del usuario (solicitud seleccionada)
        llm_result = run_llm(query=selected_request, chat_history=st.session_state.chat_local.copy())

        # Actualiza el historial local
        st.session_state.chat_local = llm_result["chat_history"]

        # Muestra el mensaje seleccionado en el historial del frontend
        st.session_state.messages.append({"role": "human", "content": selected_request})

        # Muestra la respuesta del LLM
        st.session_state.messages.append({"role": "assistant", "content": llm_result["response"]})

        # Verifica si hay fuentes para mostrar
        if llm_result.get("sources"):
            sources_markdown = "Fuentes: " + ", ".join(
                [f"[{i + 1}]({source})" for i, source in enumerate(llm_result["sources"])]
            )
            st.session_state.messages.append({"role": "assistant", "content": sources_markdown})

# Muestra los mensajes anteriores del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura el prompt del usuario y lo procesa
# Captura el prompt del usuario y lo procesa
user_prompt = st.chat_input("Pregúntale al ChatBot")
if user_prompt:
    # Muestra el mensaje del usuario inmediatamente
    with st.chat_message("human"):
        st.markdown(user_prompt)

    # Agrega el mensaje del usuario al historial
    st.session_state.messages.append({"role": "human", "content": user_prompt})

    # Procesar la consulta personalizada
    with st.spinner("Cargando respuesta..."):
        llm_result = run_llm(query=user_prompt, chat_history=st.session_state.chat_local.copy())

        # Actualiza el historial local
        st.session_state.chat_local = llm_result["chat_history"]

        # Muestra la respuesta del asistente inmediatamente
        with st.chat_message("assistant"):
            st.markdown(llm_result["response"])

        # Agrega la respuesta al historial
        st.session_state.messages.append({"role": "assistant", "content": llm_result["response"]})

        # Verifica si hay fuentes para mostrar
        if llm_result.get("sources"):
            sources_markdown = "Fuentes: " + ", ".join(
                [f"[{i + 1}]({source})" for i, source in enumerate(llm_result["sources"])]
            )
            with st.chat_message("assistant"):
                st.markdown(sources_markdown)

            # Agrega las fuentes al historial
            st.session_state.messages.append({"role": "assistant", "content": sources_markdown})
