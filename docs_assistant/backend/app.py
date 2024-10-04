import streamlit as st
from cores import run_llm  # Importamos la función run_llm desde cores.py

# Inicializamos el historial de chat en sesión
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Título del chat
st.title("Chat con LangChain y Pinecone")


# Función para mostrar el historial en el frontend
def display_chat_history(chat_history):
    for message in chat_history:
        if message["role"] == "human":
            st.markdown(f"**Usuario:** {message['content']}")
        else:
            st.markdown(f"**Asistente:** {message['content']}")


# Cuadro de texto para el input del usuario
user_input = st.text_input("Escribe tu pregunta aquí:")

# Botón para enviar la consulta
if st.button("Enviar") and user_input:
    # Ejecutamos la función run_llm con el input del usuario y el historial actual
    response_data = run_llm(query=user_input, chat_history=st.session_state['chat_history'])

    # Actualizamos el historial de chat
    st.session_state['chat_history'] = response_data["chat_history"]

    # Mostramos el historial actualizado
    display_chat_history(st.session_state['chat_history'])

    # Mostramos la respuesta y las fuentes
    st.markdown("### Respuesta del asistente:")
    st.markdown(response_data['response'])

    # Mostramos los enlaces de los documentos fuente
    st.markdown("### Fuentes:")
    for source in response_data['sources']:
        st.markdown(f"- [{source}]({source})")

# Si ya hay un historial de chat, lo mostramos
if st.session_state['chat_history']:
    display_chat_history(st.session_state['chat_history'])
