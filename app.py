from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Extract data from the pdf file

def pdfLoader(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

extracted_data = pdfLoader("Data/")

extracted_data



import streamlit as st

# Interface do Streamlit
st.title("MediBôti")

# Entrada de consulta
user_query = st.text_input("Digite sua consulta:")



