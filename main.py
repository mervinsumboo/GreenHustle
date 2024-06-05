import io

import streamlit as st
from PyPDF2 import PdfReader
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from readPDF import read_pdf_content, read_text_file

# Set assistant icon to a finance logo
icons = {"assistant": "./logo_01.svg", "user": "💼"}

# Initialize Mistral client with your API key
api_key = "ENTER YOUR API_KEY HERE"
model = "open-mixtral-8x22b"
client = MistralClient(api_key=api_key)


prompt_path = "content/prompt.txt"
prompt_content = read_text_file(prompt_path)
space = "\n\n"

# App title
st.set_page_config(page_title="fi. petit effort, GRANDE récompense")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi. I'm fi. Tell me what's your salary and all your expenses."},
    ]


# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    clear_chat_history()

# Display or clear chat messages
for message in st.session_state.messages:
    if message["role"] in ["assistant", "user"]:
        with st.chat_message(message["role"], avatar=icons[message["role"]]):
            st.write(message["content"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Process the uploaded file
    file_contents = uploaded_file.read()
    pdf_reader = PdfReader(io.BytesIO(file_contents))

    # Iterate through each page and extract text
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    st.session_state.messages.append(
        {"role": "system", "content": prompt_content + space + text})
    st.write("File uploaded successfully!")

with st.sidebar:
    st.title('fi. petit effort, GRANDE récompense')
    st.sidebar.button('Clear chat history', on_click=clear_chat_history)


# Function for generating Budget Planner response using Mistral LLM
def generate_finance_response():
    messages = []
    for message in st.session_state.messages:
        if message["role"] == "system":
            messages.append(ChatMessage(
                role="user", content=message["content"]))
        else:
            messages.append(message)
    chat_response = client.chat(
        model=model,
        messages=messages,
    )
    response_content = chat_response.choices[0].message.content
    return response_content


# User-provided promptlocal
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="💬"):
        st.write(prompt)

    # Generate a new response using Mistral LLM
    response = generate_finance_response()
    with st.chat_message("assistant", avatar="./logo_01.svg"):
        st.write(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
