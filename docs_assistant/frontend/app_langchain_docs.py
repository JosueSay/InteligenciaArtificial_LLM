import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.core_langchain_docs import run_llm

st.title("ChatBot Assistant Langchain")

# Historial que se obtiene después de cada prompt con llm
if "chat_local" not in st.session_state:
    st.session_state.chat_local = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra los mensajes anteriores del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura el prompt del usuario y lo muestra
user_prompt = st.chat_input("Preguntale al ChatBot")
if user_prompt:
    # print(f"\n==================OTRA ITERACION==================")
    with st.chat_message("human"):
        st.markdown(user_prompt)

    # Añade el mensaje del usuario al historial
    st.session_state.messages.append({"role": "human", "content": user_prompt})

    # Muestra el spinner mientras se procesa la respuesta del modelo
    with st.spinner("Cargando respuesta..."):
        # Imprimir el historial local
        # print("(FRONTEND) Historial antes de recibir respuesta:")
        # if st.session_state.chat_local:
        #     for i, message in enumerate(st.session_state.chat_local, start=1):
        #         print(f"\t{i}. {message['role']}: {message['content']}")
        # else:
        #     print("\tNo hay historial previo.")

        # Llamar a la función LLM con el input del usuario y una copia del historial de conversación
        llm_result = run_llm(query=user_prompt, chat_history=st.session_state.chat_local.copy())

        # Imprimir el historial que trae el LLM como respuesta
        # print("(FRONTEND) Historial respuesta del llm:")
        # if llm_result["chat_history"]:
        #     for i, message in enumerate(llm_result["chat_history"], start=1):
        #         print(f"\t{i}. {message['role']}: {message['content']}")
        # else:
        #     print("\tNo hay historial previo.")

        # Actualiza el historial local
        if not st.session_state.chat_local:
            st.session_state.chat_local += llm_result["chat_history"]
        else:
            st.session_state.chat_local = llm_result["chat_history"]

        # Mostrar el historial local
        # print("(FRONTEND) Historial local después de recibir respuesta llm:")
        # if st.session_state.chat_local:
        #     for i, message in enumerate(st.session_state.chat_local, start=1):
        #         print(f"\t{i}. {message['role']}: {message['content']}")
        # else:
        #     print("\tNo hay historial previo.")

        # Muestra la respuesta del LLM en el chat
        with st.chat_message("assistant"):
            st.markdown(llm_result["response"])

    # Añade la respuesta al historial de mensajes
    st.session_state.messages.append({"role": "assistant", "content": llm_result["response"]})

    # Muestra las fuentes como enlaces
    with st.chat_message("assistant"):
        sources_markdown = "Fuentes: " + ", ".join(
            [f"[{i + 1}]({source})" for i, source in enumerate(llm_result["sources"])]
        )
        st.markdown(sources_markdown)

    # Añade las fuentes al historial de mensajes
    st.session_state.messages.append({"role": "assistant", "content": sources_markdown})

    # Imprime en la consola la respuesta y las fuentes para depuración
    # print(f"(FRONTEND) Respuesta recibida de IA: {llm_result['response']}")
