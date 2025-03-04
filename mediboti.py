import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint


DB_FAISS_PATH = 'vectorstore/db_faiss'
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(repo_id, HF_TOKEN):
    llm= HuggingFaceEndpoint(repo_id=repo_id, 
                             temperature=0.5,
                             model_kwargs={'token':HF_TOKEN,
                                           'max_length': 512})
    return llm 

def main():
    st.title("MediBoti")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("ask me something")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})


        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont't know the answer, you can say "I don't know", and dont't try to make it up a response.
            Dont provide anything out of the given context.

            Context: {context}
            Question: {question}

            Start to answer the question directly no small talk please.
            """
        HUGGINGFACE_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
        HF_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
        llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("fail to load vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response=qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]
            result_to_show = result+"\nSource Docs:\n"+str(source_documents)


        except Exception as e:
            result_to_show = f"ERROR: {e}"

        st.chat_message('assistant').markdown(result_to_show)
        st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})


if __name__ == "__main__":
    main()


    
    