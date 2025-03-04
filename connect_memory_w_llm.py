import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# setup llm (mistral)
HF_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
HUGGINGFACE_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(repo_id):
    llm= HuggingFaceEndpoint(repo_id=repo_id, 
                             temperature=0.5,
                             model_kwargs={'token':HF_TOKEN,
                                           'max_length': 512})
    return llm 

# connect llm with faiss and create faiss


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont't know the answer, you can say "I don't know", and dont't try to make it up a response.
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start to answer the question directly no small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# load database 
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH,embeddings=embedding_model,allow_dangerous_deserialization=True)


# create retrieval qa
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# query

user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])